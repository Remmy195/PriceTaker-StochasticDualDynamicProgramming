# using JuMP, CSV, DataFrames, GLPK, Plots, Tables

# # Define the directory path and filename
# directory_path = raw"C:\Users\Remi\OneDrive - UCB-O365\Documents\lecture notes\MSc\Thesis\Julia\Data"
# filename = "SDDP_Hourly.csv"
# file_path = joinpath(directory_path, filename)
# data = CSV.read(file_path, DataFrames.DataFrame)

# horizon_length = 71  # Define the length of the rolling horizon (e.g., 24 hours)
# time_res = 8760  # Calculate the number of horizons
# opti_length = 24
# iterations_length = time_res/opti_length

# ############################################################################################
# # Loop for Daily Optimization with Look Ahead Window
# ############################################################################################

# # Create an empty array to store the profit results for each horizon
# profit_results = Float64[]
# SOC_data = Array[]
# initial = Float64[]
# e_charge_data = Array[]
# e_discharge_data = Array[]
# rev_data = Array[]
# cost_data = Array[]

# ini_energy = 5000
# # Rolling horizon loop for Daily Optimization
# for horizon in 1:(time_res/opti_length)
#     # Define the time indices for the current horizon
#     horizon_start = (horizon-1) * opti_length + 1
#     horizon_end = min((horizon_start + horizon_length), time_res)
#     t = horizon_start:horizon_end
#     t= Int.(t)
#     Price = data[!, :Price]
#     E_min = 0
#     p = 500 #MW
#     h = 10 #hours
#     E_max = h*p #MWh
#     PF = 0
#     eff_c = 0.52
#     eff_dc = 0.52
#     cost_op = 0
#     rate_loss = 0
    
   
#     Ptaker = Model(GLPK.Optimizer)
#     @variables(Ptaker, begin
#         e_stored[t in t]
#         e_pur[t in t]
#         e_sold[t in t]
#         e_discharge[t in t] >= 0
#         e_charge[t in t] >= 0
#         E_min <= e_aval[t in t] <= E_max
#         e_loss[t in t]
#         rev[t in t]
#         cost[t in t]
#         cost_pur[t in t]
#         z_charge[t in t], Bin
#         z_dischar[t in t], Bin
#     end)
#     @constraints(Ptaker, begin
#         [t in t], e_charge[t] <= z_charge[t] * p
#         [t in t], e_discharge[t] <= z_dischar[t] * p
#         [t in t], z_charge[t] + z_dischar[t] <= 1
#         [t in t], e_charge[t] == e_pur[t] * eff_c
#         [t in t], e_discharge[t] == e_sold[t] / eff_dc
#         [t in t], e_aval[t] == e_stored[t] - e_loss[t]
#         [t in t], e_loss[t] == e_stored[t] * rate_loss
#         [t in t], rev[t] == Price[t] * e_sold[t]  # Use price data for the current horizon
#         [t in t], cost[t] == Price[t] * e_pur[t]
#         [t in (horizon_start + 1):horizon_end], e_stored[t] == e_stored[t-1] + e_charge[t] - e_discharge[t]
#         [t in t], e_stored[horizon_start] == ini_energy + e_charge[horizon_start] - e_discharge[horizon_start]
#     end)
    
#     @objective(Ptaker, Max, (sum(rev[t] - cost[t] for t = t)))
    
#     optimize!(Ptaker)
    
#     # Store the results for the current horizon
#     push!(profit_results, objective_value(Ptaker))
#     SOC_values = value.(e_stored)
#     push!(SOC_data, SOC_values)
#     ini_SOC_values = value.(e_stored)[horizon_start+(opti_length-1)]
#     push!(initial, ini_SOC_values)
#     e_charge_values = value.(e_charge)
#     push!(e_charge_data, e_charge_values)
#     e_discharge_values = value.(e_discharge)
#     push!(e_discharge_data, e_discharge_values)
#     rev_values = value.(rev)
#     push!(rev_data, rev_values)
#     cost_values = value.(cost)
#     push!(cost_data, cost_values)


#     if horizon_end == time_res
#         break  # Stop the loop when horizon_end reaches 8760
#     end
#     # Update the final e_stored value for the next iteration
#     global ini_energy = value.(e_stored)[horizon_start+(opti_length-1)]
    
# end

# # Create a DataFrame to store the variable results
# results_df = DataFrame(
#     Profit = profit_results,
#     Initial_Energy = initial,
#     SOC = SOC_data,
#     Charge_Energy = e_charge_data,
#     Discharge_Ebergy = e_discharge_data,
#     Revenue = rev_data,
#     Cost = cost_data
# )

# # Define the directory path and filename for the results CSV
# results_filename = "results.csv"
# results_file_path = joinpath(directory_path, results_filename)

# # Write the DataFrame to a CSV file
# CSV.write(results_file_path, results_df)


using JuMP, CSV, DataFrames, GLPK, Plots, Tables


############################################################################################
# Daily Optimization with Look Ahead Window
############################################################################################


# Function to define the model and solve it for a given horizon
function optimize_horizon(horizon_start::Int, horizon_end::Int, Price, ini_energy, E_min, E_max, PF, eff_c, eff_dc, cost_op, rate_loss, p)
    Ptaker = Model(GLPK.Optimizer)
    
    @variables(Ptaker, begin
        e_stored[t = horizon_start:horizon_end]
        e_pur[t = horizon_start:horizon_end]
        e_sold[t = horizon_start:horizon_end]
        e_discharge[t = horizon_start:horizon_end] >= 0
        e_charge[t = horizon_start:horizon_end] >= 0
        E_min <= e_aval[t = horizon_start:horizon_end] <= E_max
        e_loss[t = horizon_start:horizon_end]
        rev[t = horizon_start:horizon_end]
        cost[t = horizon_start:horizon_end]
        cost_pur[t = horizon_start:horizon_end]
        z_charge[t = horizon_start:horizon_end], Bin
        z_dischar[t = horizon_start:horizon_end], Bin
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
        [t = horizon_start + 1:horizon_end], e_stored[t] == e_stored[t - 1] + e_charge[t] - e_discharge[t]
        e_stored[horizon_start] == ini_energy + e_charge[horizon_start] - e_discharge[horizon_start]
    end)

    @objective(Ptaker, Max, sum(rev[t] - cost[t] for t = horizon_start:horizon_end))
    
    optimize!(Ptaker)
    
    return Ptaker, value.(e_stored), value.(e_charge), value.(e_discharge), value.(rev), value.(cost)
end

# Function to run the rolling horizon optimization
function run_rolling_horizon_optimization(data, horizon_length, time_res, opti_length, ini_energy)
    # Define constants
    E_min = 0
    p = 500
    h = 10
    E_max = h * p
    PF = 0
    eff_c = 0.52
    eff_dc = 0.52
    cost_op = 0
    rate_loss = 0
    

    # Create arrays to store results
    profit_results = Float64[]
    SOC_data = Vector{Vector{Float64}}()
    initial = Float64[]
    e_charge_data = Vector{Vector{Float64}}()
    e_discharge_data = Vector{Vector{Float64}}()
    rev_data = Vector{Vector{Float64}}()
    cost_data = Vector{Vector{Float64}}()



    # Rolling horizon loop for Daily Optimization
    for horizon in 1:(time_res / opti_length)
        horizon_start = (horizon - 1) * opti_length + 1
        horizon_end = min((horizon_start + horizon_length), time_res)
        t = horizon_start:horizon_end
        t = Int.(t)
        Price = data[!, :Price]
        
        # Call the optimization function
        Ptaker, SOC_values, e_charge_values, e_discharge_values, rev_values, cost_values =
            optimize_horizon(Int(horizon_start), Int(horizon_end), Price, ini_energy, E_min, E_max, PF, eff_c, eff_dc, cost_op, rate_loss, p)

        # Store results
        push!(profit_results, objective_value(Ptaker))
        push!(SOC_data, SOC_values)
        push!(initial, SOC_values[Int(horizon_start+(opti_length-1))])
        push!(e_charge_data, e_charge_values)
        push!(e_discharge_data, e_discharge_values)
        push!(rev_data, rev_values)
        push!(cost_data, cost_values)

        if horizon_end == time_res
            break
        end

        global ini_energy = SOC_values[Int(horizon_start+(opti_length-1))]
    end

    # Create a DataFrame to store the variable results
    results_df = DataFrame(
        Profit = profit_results,
        Initial_Energy = initial,
        SOC = SOC_data,
        Charge_Energy = e_charge_data,
        Discharge_Ebergy = e_discharge_data,
        Revenue = rev_data,
        Cost = cost_data
    )

    return results_df
end

# Define the directory path and filename
directory_path = raw"C:\Users\Remi\OneDrive - UCB-O365\Documents\lecture notes\MSc\Thesis\Julia\Data"
filename = "SDDP_Hourly.csv"
file_path = joinpath(directory_path, filename)
data = CSV.read(file_path, DataFrames.DataFrame)

horizon_length = 71
time_res = 8760
opti_length = 24
ini_energy = 5000

results_df = run_rolling_horizon_optimization(data, horizon_length, time_res, opti_length, ini_energy)

# Define the directory path and filename for the results CSV
results_filename = "results.csv"
results_file_path = joinpath(directory_path, results_filename)

# Write the DataFrame to a CSV file
CSV.write(results_file_path, results_df)
