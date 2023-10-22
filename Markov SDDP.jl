
using Base.Threads, CSV, DataFrames, CPLEX, Plots, SDDP, Base.Filesystem, BenchmarkTools


# Define the directory path and filename
directory_path = raw"C:\Users\Remi\OneDrive - UCB-O365\Documents\lecture notes\MSc\Thesis\Julia\Data"
filename = "residuals_data.csv"
file_path = joinpath(directory_path, filename)
data = CSV.read(file_path, DataFrames.DataFrame)


# #Function to generate price scenerios (Monte Carlo Simulation)
# function simulator()
#     μ, α = 0, 12.9116813709803
#     num_samples = 8760
#     samples = μ .+ α * randn(num_samples)
#     noise = samples
#     ε = collect(Float64, noise)
#     VRE = data.VRE
#     Demand = data.Demand
#     price = zeros(8760)  # Initialize array to store scenarios
    
#     for t in 1:8760
#         # Access specific row of data for this iteration
#         reg = (0.000202214130177962 * Demand[t]) + (-0.000831712624823633 * VRE[t]) + 27.4180446249749 + rand(ε)
#         price[t] = reg  # Store the generated price in the scenario array
#     end
#     return price
# end


#Function to generate price scenerios (Monte Carlo Simulation)
function simulator()
    μ, α = 0, 10.1
    num_samples = 8760
    samples = μ .+ α * randn(num_samples)
    noise = samples
    ε = collect(Float64, noise)
    VRE = data.VRE
    Demand = data.Demand
    Trend = data.Trend
    price = zeros(8760)  # Initialize array to store scenarios
    
    for t in 1:8760
        # Access specific row of data for this iteration
        reg = (0.000126392446876915 * Demand[t]) + (-0.000565009085257919 * VRE[t]) + (0.975145474763468 * Trend[t]) + 7.86672219533408 + rand(ε)
        price[t] = reg  # Store the generated price in the scenario array
    end
    return price
end


μ, α = 0, 8.74308483747609
data = Float64[]  # Initialize an empty array to store data

# Create a color palette for the histogram
# colors = range(HSV(0, 1, 1), stop=HSV(-360, 1, 1), length=90)

anim = @animate for i=1:8760
    p = μ .+ α * randn(1)  # Generate a single random data point
    push!(data, p[1])  # Store the data point in the array
    histogram(data, legend=false)
end

gif(anim, fps = 6)





# @btime simulator()

# Generate price scenarios
num_scenarios = 100
scenarios = [simulator() for _ in 1:num_scenarios]

# Create animation plot
anim = Animation()
p = plot(1:8760, scenarios[1], label="Scenario 1", xlim=(1, 8760), ylim=(minimum(scenarios)..., maximum(scenarios)...))
for i in 2:num_scenarios
    plot!(p, 1:8760, scenarios[i], label="Scenario $i")
    frame(anim, p)
end
gif(anim, "animation.gif", fps = 5)



# Extract price realizations for each stage
price_realizations = hcat(scenarios...)


   #=Static plot for price
p_plot = Plots.plot(
    [simulator() for _ in 1:10000];
    legend = false,
    xlabel = "hour",
    ylabel = "Price [\$/MW]",
    )=#

#Create Markovian Graph
graph = SDDP.MarkovianGraph(simulator; budget = 8760, scenarios = num_scenarios)

p = plot()
for ((t, price), edges) in graph.nodes
    for ((t′, price′), probability) in edges
        plot!(
            p,
            [t, t′],
            [price, price′];
            color = "red",
            width = 2 * probability,
        )
    end
end

# Display the plot
display(p)







#Model
Ptaker = SDDP.PolicyGraph(
    graph,
    sense = :Max,
    upper_bound = 100000,
    optimizer = CPLEX.Optimizer,
) do sp, node
    t, price = node
    E_min = 0
    p = 500 #MW
    h = 10 #hours
    E_max = h*p #MWh
    PF = 0
    eff_c = 0.52
    eff_dc = 0.52
    cost_op = 0
    rate_loss = 0
    ini_energy = 5000

    #state variable
    @variable(
        sp,
        0 <= e_stored <= E_max,
        SDDP.State,
        initial_value = ini_energy,
    )

    @variables(
        sp, begin
        e_pur
        e_sold
        e_discharge >= 0
        e_charge >= 0
        E_min <= e_aval <= E_max
        e_loss
        rev
        cost
        cost_pur
        z_charge, Bin
        z_dischar, Bin
    end
    )
    @constraints(
    sp, begin
    e_charge <= (z_charge * p)
    e_discharge <= (z_dischar * p)
    z_charge + z_dischar <= 1
    e_charge == e_pur*eff_c
    e_discharge == (e_sold/eff_dc)
    e_aval == (e_stored.out-e_loss)
    end
    )
    #Transiton Matrix and Constraints
    @constraints(
    sp, begin
    e_loss == (e_stored.out*rate_loss)
    rev == price * e_sold
    cost == (cost_pur+(cost_op*e_aval))
    cost_pur == (e_pur * price)
    e_stored.out == e_stored.in + e_charge - e_discharge
    end
    )
    @stageobjective(sp, sum(rev-cost))
end

#Train Model
SDDP.train(
    Ptaker;
    iteration_limit = 5,
)

#Simulate Model
n_replications = 2
simulations = SDDP.simulate(
    Ptaker,
    n_replications,
    Symbol[:e_stored, :e_charge, :e_discharge, :z_charge, :z_dischar, :rev, :cost],
)
length.(simulations)


replication = 2
stage = 2
simulations[replication][stage]



sim_price = [stage[:node_index] for stage in simulations[1]]

noise = [stage[:noise_term] for stage in simulations[1]]

#Static plot for price
p_plot = Plots.plot(
    sim_price;
    legend = false,
    xlabel = "hour",
    ylabel = "Price [\$/MW]",
    )


#=
Create animation
anim = Animation()

    p = plot(1:744, price_realizations[:, 1], label="Price Realization 1", xlabel="Hour", ylabel="Price", ylim=(minimum(price_realizations)..., maximum(price_realizations)...))
    plot!(p, 1:744, [sim[:e_stored].out for sim in simulations[1]], label="State of Charge 1", linestyle=:dash)
    frame(anim, p)

for i in 2:n_replications
    # Update the existing plot with new data for each iteration
    plot!(p, 1:744, price_realizations[:, 100], label="Price Realization $i")
    plot!(p, 1:744, [sim[:e_stored].out for sim in simulations[i]], label="State of Charge $i", linestyle=:dash)
    frame(anim, p)
end

gif(anim, "price_soc_simulation_animation.gif", fps = 10)
=#


# Create animation
anim = Animation()

chart = plot(1:744, [sim[:e_stored].out for sim in simulations[1]], label="State of Charge 1", linestyle=:dash)
frame(anim, chart)
for i in 2:n_replications
    plot!(chart, 1:744, [sim[:e_stored].out for sim in simulations[i]], label="State of Charge $i", linestyle=:dash)
    frame(anim, chart)
end

gif(anim, "d_animation.gif", fps = 10)













#Plots
chart = SDDP.SpaghettiPlot(simulations)
SDDP.add_spaghetti(chart; title = "Storage") do sim
    return sim[:e_stored].out
end
SDDP.add_spaghetti(chart; title = "Charge Energy") do sim
    return sim[:e_charge]
end
SDDP.add_spaghetti(chart; title = "Discharge Energy") do sim
    return sim[:e_discharge]
end
SDDP.add_spaghetti(chart; title = "Charge Decision") do sim
    return sim[:z_charge]
end
SDDP.add_spaghetti(chart; title = "Discharge Decision") do sim
    return sim[:z_dischar]
end
SDDP.add_spaghetti(chart; title = "Revenue") do sim
    return sim[:rev]
end
SDDP.add_spaghetti(chart; title = "Cost") do sim
    return sim[:cost]
end
SDDP.plot(
    chart,
    "spaghetti_plot.html";
    # To open graph in browser
    open = true,
)

SDDP.publication_plot(simulations; title = "Storage") do sim
    return sim[:e_stored].out
end
SDDP.publication_plot(simulations; title = "Charge Energy") do sim
    return sim[:e_charge]
end
SDDP.publication_plot(simulations; title = "Discharge Energy") do sim
    return sim[:e_discharge]
end
SDDP.publication_plot(simulations; title = "Charge Decision") do sim
    return sim[:z_charge]
end
SDDP.publication_plot(simulations; title = "Discharge Decision") do sim
    return sim[:z_dischar]
end
objectives = map(simulations) do simulation
    return sum(stage[:stage_objective] for stage in simulation)
end

μ, ci = SDDP.confidence_interval(objectives)
println("Confidence interval: ", μ, " ± ", ci);
println("Lower bound: ", SDDP.calculate_bound(Ptaker));



