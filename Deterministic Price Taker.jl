# using JuMP, CSV, DataFrames, HiGHS, Plots

# Ptaker = Model(HiGHS.Optimizer)

# # Define the directory path and filename
# directory_path = raw"C:\Users\Remi\OneDrive - UCB-O365\Documents\lecture notes\MSc\Thesis\Julia\Data"
# filename = "SDDP_Daily.csv"
# file_path = joinpath(directory_path, filename)
# data = CSV.read(file_path, DataFrames.DataFrame)

# time = size(data, 1)
# t = 1:time
# pr = data[t, :Price]
# plot(pr, label="Deterministic Price", xlabel = "Timestep", ylabel = "Price")
# savefig("Determinstic price.png")

# E_min = 0
# p = 500 #MW
# h = 10 #hours
# E_max = h*p #MWh
# PF = 0
# eff_c = 0.52
# eff_dc = 0.52
# cost_op = 0
# rate_loss = 0
# ini_energy = 5000 

# @variables(Ptaker, begin
#     e_stored[t]
#     e_pur[t]
#     e_sold[t]
#     e_discharge[t] >= 0
#     e_charge[t] >= 0
#     E_min <= e_aval[t] <= E_max
#     e_loss[t]
#     rev[t]
#     cost[t]
#     cost_pur[t]
#     Profit
#     z_charge[t], Bin
#     z_dischar[t], Bin
# end
#     )
# @constraints(Ptaker, begin
#     [t in t], e_charge[t] <= z_charge[t] * p
#     [t in t], e_discharge[t] <= z_dischar[t] * p
#     [t in t], z_charge[t] + z_dischar[t] <= 1
#     [t in t], e_charge[t] == e_pur[t] * eff_c
#     [t in t], e_discharge[t] == e_sold[t] / eff_dc
#     [t in t], e_aval[t] == e_stored[t] - e_loss[t]
#     [t in t], e_loss[t] == e_stored[t] * rate_loss
#     [t in t], rev[t] == pr[t] * e_sold[t]
#     [t in t], cost[t] == cost_pur[t] + (cost_op*e_aval[t])
#     [t in t], cost_pur[t] == (e_pur[t] * pr[t])
#     [t in 2:time], e_stored[t] == e_stored[t-1] + e_charge[t] - e_discharge[t]
#     e_stored[1] == ini_energy + e_charge[1] - e_discharge[1]
# end
# )
# @objective(Ptaker, Max, (sum(rev[t] - cost[t] for t in t)))
# optimize!(Ptaker)
# solution_summary(Ptaker)
# objective_value(Ptaker)
# display(value.(e_stored))
# SOC = collect(value.(e_stored))
# plot(SOC)
# # Create an empty array to store normalized_SOC values
# normalized_SOC_values = Float64[]

# # Calculate normalized SOC values
# for i in SOC
#     normalized_SOC = (i - 0) / (E_max - 0)
#     push!(normalized_SOC_values, normalized_SOC)  # Store the normalized value in the array
#     println(normalized_SOC)
# end


# #Static plot for price
# p_plot = plot(
#     t, normalized_SOC_values;  # Provide t as the x-axis values
#     legend = true,
#     xlabel = "hour",
#     ylabel = "SOC [MWh]",
# )

# Plots.savefig("DSOC.png")

using JuMP, CSV, DataFrames, GLPK, Plots

function optimize_price_deterministic(data)
    Ptaker = Model(GLPK.Optimizer)

    time = size(data, 1)
    t = 1:time
    pr = data[t, :Price]
    E_min = 0
    p = 500 #MW
    h = 10 #hours
    E_max = h * p #MWh
    PF = 0
    eff_c = 0.52
    eff_dc = 0.52
    cost_op = 0
    rate_loss = 0
    ini_energy = 5000 

    @variables(Ptaker, begin
        e_stored[t]
        e_pur[t]
        e_sold[t]
        e_discharge[t] >= 0
        e_charge[t] >= 0
        E_min <= e_aval[t] <= E_max
        e_loss[t]
        rev[t]
        cost[t]
        cost_pur[t]
        Profit
        z_charge[t], Bin
        z_dischar[t], Bin
    end)

    @constraints(Ptaker, begin
        [t in t], e_charge[t] <= z_charge[t] * p
        [t in t], e_discharge[t] <= z_dischar[t] * p
        [t in t], z_charge[t] + z_dischar[t] <= 1
        [t in t], e_charge[t] == e_pur[t] * eff_c
        [t in t], e_discharge[t] == e_sold[t] / eff_dc
        [t in t], e_aval[t] == e_stored[t] - e_loss[t]
        [t in t], e_loss[t] == e_stored[t] * rate_loss
        [t in t], rev[t] == pr[t] * e_sold[t]
        [t in t], cost[t] == cost_pur[t] + (cost_op * e_aval[t])
        [t in t], cost_pur[t] == (e_pur[t] * pr[t])
        [t in 2:time], e_stored[t] == e_stored[t-1] + e_charge[t] - e_discharge[t]
        e_stored[1] == ini_energy + e_charge[1] - e_discharge[1]
    end)

    @objective(Ptaker, Max, (sum(rev[t] - cost[t] for t in t)))
    optimize!(Ptaker)
    solution_summary(Ptaker)
    objective_val = objective_value(Ptaker)
    e_stored_values = collect(value.(e_stored))

    return e_stored_values, objective_val
end

function normalize_SOC(e_stored_values, E_max)
    normalized_SOC_values = [(i - 0) / (E_max - 0) for i in e_stored_values]
    return normalized_SOC_values
end

function plot_normalized_SOC(t, normalized_SOC_values)
    p_plot = plot(
        t, normalized_SOC_values;
        legend = true,
        xlabel = "hour",
        ylabel = "Normalized SOC",
    )
    return p_plot
end

# Define the directory path and filename
directory_path = raw"C:\Users\Remi\OneDrive - UCB-O365\Documents\lecture notes\MSc\Thesis\Julia\Data"
filename = "SDDP_Hourly.csv"
file_path = joinpath(directory_path, filename)
data = CSV.read(file_path, DataFrames.DataFrame)

e_stored_values, objective_val = optimize_price_deterministic(data)
E_max = 5000 # Replace with the actual value
normalized_SOC_values = normalize_SOC(e_stored_values, E_max)
p_plot = plot_normalized_SOC(1:size(data, 1), normalized_SOC_values)
