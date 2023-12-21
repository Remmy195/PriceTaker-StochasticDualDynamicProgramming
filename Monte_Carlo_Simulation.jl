#############################################################################################################
#############################################################################################################
# Optimization of Long-Duration Energy Storage Price Taker - Monte Carlo Deterministic Optimal Control Model 
#############################################################################################################
#############################################################################################################

# Load Packages
using JuMP, CSV, DataFrames, Gurobi, Plots, BenchmarkTools, Distributions, XLSX, Tables
const GRB_ENV = Gurobi.Env()

# Function Definitions
# Optimal Control Model
function optimize_price_deterministic(data, h, price)
    Ptaker = Model(() -> Gurobi.Optimizer(GRB_ENV))
    T = size(data, 1) #Optimization horizon
    #price = data[1:T, :Price] #Energy price
    E_min = 0 #Energy Minimum MWh
    p = 1000 # Power in MW
    E_max = h*p #Energy Minimum MWh
    PF = 0 #Power Factor
    eff_c = 0.725 # Charge Efficiency
    eff_dc = 0.507 #Discharge Efficiency
    cost_op = 0 # Operational Cost for each hour of Operation
    rate_loss = 0 # Self Discharge rate 
    ini_energy = E_max #Initial State of Charge

    @variables(Ptaker, begin
        E_min <= e_stored[1:T+1] <= E_max #State of charge
        e_pur[1:T] #Energy purchased
        e_sold[1:T] #Energy sold
        e_discharge[1:T] >= 0 #Energy discharged from storage
        e_charge[1:T] >= 0 #Energy Charged into storage
        E_min <= e_aval[1:T] <= E_max #Energy available after self discharge
        e_loss[1:T] #Energy lost due to self discharge
        rev[1:T] # Revenue from energy sold to grid
        cost_pur[1:T] # Cost of energy bought from grid
        cost[1:T] # Total Cost
        profit[1:T] #Proft from arbitrage operation
        z_charge[1:T], Bin # Charge Decision
        z_dischar[1:T], Bin # Discharge Decision
    end
    )

    fix(e_stored[1], ini_energy; force = true)

    @constraints(Ptaker, begin
        [t in 1:T], e_charge[t] <= z_charge[t] * p
        [t in 1:T], e_discharge[t] <= z_dischar[t] * p
        [t in 1:T], z_charge[t] + z_dischar[t] <= 1
        [t in 1:T], e_charge[t] == e_pur[t] * eff_c
        [t in 1:T], e_discharge[t] == e_sold[t] / eff_dc
        [t in 1:T], e_aval[t] == e_stored[t] - e_loss[t]
        [t in 1:T], e_loss[t] == e_stored[t] * rate_loss
        [t in 1:T], rev[t] == price[t] * e_sold[t]
        [t in 1:T], cost[t] == cost_pur[t] + (cost_op*e_aval[t])
        [t in 1:T], cost_pur[t] == e_pur[t] * price[t]
        [t in 1:T], e_stored[t+1] == e_stored[t] + e_charge[t] - e_discharge[t]
        [t in 1:T], profit[t] == rev[t] - cost[t]
    end
    )

    @objective(Ptaker, Max, (sum(profit[t] for t in 1:T)))
    optimize!(Ptaker)

    if termination_status(Ptaker) == MOI.OPTIMAL
        # Collection of the results can proceed here
    else
        # Handle the case where optimization did not succeed
        println("Optimization was not successful. Status: ", termination_status(Ptaker))
    end

    #Collect Results
    summary = solution_summary(Ptaker)
    objective_val = objective_value(Ptaker)
    e_stored_values = collect(value.(e_stored))
    e_charge_values = collect(value.(e_charge))
    e_discharge_values = collect(value.(e_discharge))
    e_pur_values = collect(value.(e_pur))
    e_sold_values = collect(value.(e_sold))
    profit_values = collect(value.(profit))
    z_charge_values = collect(value.(z_charge))
    z_dischar_values = collect(value.(z_dischar))
    e_loss_values = collect(value.(e_loss))
    e_aval_values = collect(value.(e_aval))
    revenue_values = collect(value.(rev))
    cost_values = collect(value.(cost))
    cost_pur_values = collect(value.(cost_pur))

    # Create a DataFrame to store the results
    results_df = DataFrame(
        e_stored = e_stored_values,
        e_charge = [e_charge_values; missing],
        e_discharge = [e_discharge_values; missing],
        e_pur = [e_pur_values; missing],
        e_sold = [e_sold_values; missing],
        profit = [profit_values; missing],
        z_charge = [z_charge_values; missing],
        z_discharge = [z_dischar_values; missing],
        e_loss = [e_loss_values; missing],
        e_aval = [e_aval_values; missing],
        revenue = [revenue_values; missing],
        cost = [cost_values; missing],
        cost_pur = [cost_pur_values; missing],
    )


    return results_df, objective_val, summary
end



# Scenario Generation Function
function simulate_prices(data, method_type)
    price = zeros(8760)

    if method_type == :t_distribution
        # Student's t-distribution parameters
        df = 4.30  # Degrees of Freedom
        loc = 0.51 # Location
        scale = 6.16 # Scale
        
        for t in 1:8760
            ε = rand(TDist(df), 1)[1] * scale + loc
            price[t] = calculate_price(data, t) + ε
        end
    elseif method_type == :normal_distribution
        # Normal distribution parameters
        μ, α = 0, 9.05372307007077 # Mean, SD 

        for t in 1:8760
            ε = μ + α * randn()
            price[t] = calculate_price(data, t) + ε
        end
    else
        error("Invalid method type")
    end

    return price
end

function calculate_price(data, t)
    return -10.5073 * data.S1[t] +  7.1459 * data.C1[t] +
           2.9938 * data.S2[t] +  3.6828 * data.C2[t] +
           0.6557 * data.S3[t] +  1.9710 * data.C3[t] + 
           0.9151 * data.S4[t] + -1.4882 * data.C4[t] + 
           0.0000705481036188987 * data.Load[t] +
           -0.000673131618513161 * data.VRE[t] +
           25.6773336136185
end

###################################################################################################################################
# Main Script Execution
###################################################################################################################################
# Define the directory path and filename
data = CSV.read("/projects/reak4480/Documents/SDDP_Hourly.csv", DataFrames.DataFrame)

#Distribution
method_type = :normal_distribution #or pass :t_distribution if using a student t distribution

# Define the different scenarios for discharge duration 'h'
h_values = [24]#, 48, 168, 336, 720]
loop = 2

for h in h_values
    # Initialize a dictionary to hold results for each column for this 'h' value
    results_dict = Dict{String, DataFrame}()

    i = 1  # Initialize the iteration counter
    while i <= loop
        price = simulate_prices(data, method_type)
        results_df, objective_val, summary = optimize_price_deterministic(data, h, price)
        print(summary)

        # Loop over each column in results_df and store data in results_dict
        for col in names(results_df)
            # Initialize DataFrame for this column if it doesn't exist
            if !haskey(results_dict, col)
                results_dict[col] = DataFrame()
            end
            # Add iteration data as a new column
            results_dict[col][!, "Iteration_$i"] = results_df[!, col]
        end

        println("Objective Value for h=$(h): ", objective_val)
        i += 1  # Increment iteration counter
    end

    # Write the data to an Excel file for this 'h' value
    excel_filename = "/projects/reak4480/Documents/Results/Monte_Carlo/test_Deterministic_results_h$(h).xlsx"
    XLSX.openxlsx(excel_filename, mode="w") do xf
        for (col_name, df) in results_dict
            # Create or get a worksheet named after the column
            sheetname = col_name  # Using the column name as the sheet name
            sheet = XLSX.addsheet!(xf, sheetname)
    
            # Write DataFrame to the sheet
            XLSX.writetable!(sheet, df, anchor_cell=XLSX.CellRef("A1"))
        end
    end
end
