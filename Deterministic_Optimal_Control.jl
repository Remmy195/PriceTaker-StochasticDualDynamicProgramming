#######################################################################################################
#######################################################################################################
# Optimization of Long-Duration Energy Storage Price Taker Using A Deterministic Optimal Control Model 
#######################################################################################################
#######################################################################################################

# Load Packages
using JuMP, CSV, DataFrames, Gurobi, Plots, BenchmarkTools
const GRB_ENV = Gurobi.Env() # This setup reuses a single Gurobi environment (GRB_ENV) for multiple model solves by passing the GRB_ENV object to Gurobi.Optimizer

# Function Definitions
# Optimal Control Model
function optimize_price_deterministic(data, h)
    Ptaker = Model(() -> Gurobi.Optimizer(GRB_ENV))
    T = size(data, 1) #Optimization horizon
    price = data[1:T, :Price] #Energy price
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

###################################################################################################################################
# Main Script Execution
###################################################################################################################################
# Define the directory path and filename
data = CSV.read("SDDP_Hourly.csv", DataFrames.DataFrame)

# Define the different scenarios for discharge duration 'h'
h_values = [24, 48, 168, 336, 720]

# Loop through each scenario and call the optimization function
for h in h_values
    results_df, objective_val, summary = optimize_price_deterministic(data, h)
    print(summary)

    # Write the DataFrame to a CSV file with an identifier for the 'h' value
    CSV.write("Deterministic_results_h$(h).csv", results_df)

    # You can also view the objective value and the DataFrame
    println("Objective Value for h=$(h): ", objective_val)
    first(results_df, 5)  # Display the first 5 rows of the DataFrame
end
###################################################################################################################################