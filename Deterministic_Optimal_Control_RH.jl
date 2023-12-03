##########################################################################################################################
##########################################################################################################################
# Rolling Horizon Optimization of Long-Duration Energy Storage Price Taker Using A Deterministic Optimal Control Model 
##########################################################################################################################
##########################################################################################################################

# Load Packages
using JuMP, CSV, DataFrames, Gurobi, Plots, Tables, BenchmarkTools
const GRB_ENV = Gurobi.Env() # This setup reuses a single Gurobi environment (GRB_ENV) for multiple model solves by passing the GRB_ENV object to Gurobi.Optimizer

# Function Definitions
# Function to define the model and solve it for a given horizon
function optimize_horizon(horizon_start::Int, horizon_end::Int, Price, ini_energy, E_min, E_max, PF, eff_c, eff_dc, rate_loss, p)
    Ptaker = Model(() -> Gurobi.Optimizer(GRB_ENV))
    
    @variables(Ptaker, begin
        E_min <= e_stored[t = horizon_start:horizon_end+1] <= E_max #State of charge
        e_pur[t = horizon_start:horizon_end] #Energy purchased
        e_sold[t = horizon_start:horizon_end] #Energy sold
        e_discharge[t = horizon_start:horizon_end] >= 0 #Energy discharged from storage
        e_charge[t = horizon_start:horizon_end] >= 0 #Energy Charged into storage
        E_min <= e_aval[t = horizon_start:horizon_end] <= E_max #Energy available after self discharge/loss
        e_loss[t = horizon_start:horizon_end] #Energy lost due to self discharge or losses
        rev[t = horizon_start:horizon_end] # Revenue from energy sold to grid
        cost[t = horizon_start:horizon_end] # Cost of energy bought from grid
        z_charge[t = horizon_start:horizon_end], Bin # Charge Decision
        z_dischar[t = horizon_start:horizon_end], Bin # Discharge Decision
        profit[horizon_start:horizon_end] #Proft from arbitrage operation
    end)

    @constraints(Ptaker, begin
        [t = horizon_start:horizon_end], e_charge[t] <= z_charge[t] * p
        [t = horizon_start:horizon_end], e_discharge[t] <= z_dischar[t] * p
        [t = horizon_start:horizon_end], z_charge[t] + z_dischar[t] <= 1
        [t = horizon_start:horizon_end], e_charge[t] == e_pur[t] * eff_c
        [t = horizon_start:horizon_end], e_discharge[t] == e_sold[t] / eff_dc
        [t = horizon_start:horizon_end], e_aval[t] == e_stored[t] - e_loss[t]
        [t = horizon_start:horizon_end], e_loss[t] == e_stored[t] * rate_loss
        [t = horizon_start:horizon_end], rev[t] == Price[t] * e_sold[t]
        [t = horizon_start:horizon_end], cost[t] == Price[t] * e_pur[t]
        [t = horizon_start:horizon_end], e_stored[t+1] == e_stored[t] + e_charge[t] - e_discharge[t]
        e_stored[horizon_start] == ini_energy
        [t = horizon_start:horizon_end], profit[t] == rev[t] - cost[t]
    end)

    @objective(Ptaker, Max, (sum(profit[t] for t in horizon_start:horizon_end)))
    
    optimize!(Ptaker)
    if termination_status(Ptaker) == MOI.OPTIMAL
        # Extraction of the values can proceed here
    else
        # Handle the case where optimization did not succeed
        println("Optimization was not successful. Status: ", termination_status(Ptaker))
    end
    return Ptaker, value.(e_stored), value.(e_charge), value.(e_discharge), value.(rev), value.(cost), value.(z_charge), value.(z_dischar), value.(e_pur), value.(e_sold), value.(e_aval)
end

# Function to run the rolling horizon optimization
function run_rolling_horizon_optimization(data, horizon_length, time_res, opti_length, h)
    
    # Define constants
    E_min = 0 #Energy Minimum MWh
    p = 1000 # Power Output in MW
    E_max = h * p #Energy Maximum MWh
    PF = 0 #Power Factor
    eff_c = 0.725 # Charge Efficiency
    eff_dc = 0.507 #Discharge Efficiency
    rate_loss = 0 # Self Discharge rate 
    ini_energy = E_max #Initial State of Charge

    # Create arrays to store results
    profit_results = Float64[]
    SOC_data = Vector{Vector{Float64}}()
    initial = Float64[]
    e_charge_data = Vector{Vector{Float64}}()
    e_discharge_data = Vector{Vector{Float64}}()
    rev_data = Vector{Vector{Float64}}()
    cost_data = Vector{Vector{Float64}}()
    z_charge_data = Vector{Vector{Float64}}()
    z_dischar_data = Vector{Vector{Float64}}()
    e_pur_data = Vector{Vector{Float64}}()
    e_sold_data = Vector{Vector{Float64}}()
    e_aval_data = Vector{Vector{Float64}}()



    # Rolling horizon loop for Daily Optimization
    for horizon in 1:(time_res / opti_length)
        horizon_start = (horizon - 1) * opti_length + 1
        horizon_end = min((horizon_start + horizon_length), time_res)
        t = horizon_start:horizon_end
        t = Int.(t)
        Price = data[!, :Price]
        
        # Call the optimization function
        Ptaker, SOC_values, e_charge_values, e_discharge_values, rev_values, cost_values, z_charge_values, z_dischar_values, e_pur_values, e_sold_values, e_aval_values =
            optimize_horizon(Int(horizon_start), Int(horizon_end), Price, ini_energy, E_min, E_max, PF, eff_c, eff_dc, rate_loss, p)

        # Store results
        push!(profit_results, objective_value(Ptaker))
        push!(SOC_data, SOC_values)
        push!(initial, SOC_values[Int(horizon_start+opti_length)])
        push!(e_charge_data, e_charge_values)
        push!(e_discharge_data, e_discharge_values)
        push!(rev_data, rev_values)
        push!(cost_data, cost_values)
        push!(z_charge_data, z_charge_values)
        push!(z_dischar_data, z_dischar_values)
        push!(e_pur_data, e_pur_values)
        push!(e_sold_data, e_sold_values)
        push!(e_aval_data, e_aval_values)

        ini_energy = SOC_values[Int(horizon_start+opti_length)] #Update Initial State of Charge From Lookahead

    end

    # Create a DataFrame to store the variable results
    results_df = DataFrame(
        Profit = profit_results,
        Initial_Energy = initial,
        SOC = SOC_data,
        Charge_Energy = e_charge_data,
        Discharge_Energy = e_discharge_data,
        Revenue = rev_data,
        Cost = cost_data,
        Charge_Decision = z_charge_data,
        Disharge_Decision = z_dischar_data,
        Energy_Bought = e_pur_data,
        Energy_Sold = e_sold_data,
        Energy_Available = e_aval_data,
    )

    return results_df
end

###################################################################################################################################
# Main Script Execution
###################################################################################################################################
# Define the directory path and filename
data = CSV.read("SDDP_Hourly.csv", DataFrames.DataFrame)

# Define parameters for the rolling horizon
horizon_length_values = [23, 47, 71, 167, 719] # Rolling periods
time_res = 8760
opti_length = 24


# Different scenarios for discharge duration 'h'
h_values = [24, 48, 168, 336, 720]

# Run the model for different discharge durations and rolling periods
for h in h_values
    for horizon_length in horizon_length_values
        results_df = run_rolling_horizon_optimization(data, horizon_length, time_res, opti_length, h)
        # Write the DataFrame to a CSV file
        CSV.write("Deterministic_RH($horizon_length)_results_h$h.csv", results_df)
    end
end
###################################################################################################################################