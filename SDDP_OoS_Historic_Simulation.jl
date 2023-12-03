###########################################################################################
###########################################################################################
    #Long-Duration Energy Storage Price Taker Model Using Stochastic Dual Dynamic Model
###########################################################################################
###########################################################################################

# Load Packages
using CSV, DataFrames, Gurobi, Plots, SDDP, Base.Filesystem, Clustering, StatsPlots
const GRB_ENV = Gurobi.Env() # This setup reuses a single Gurobi environment (GRB_ENV) for multiple model solves by passing the GRB_ENV object to Gurobi.Optimizer

# Function Definitions
# Data Loading Function
function load_data(filepath)
    return CSV.read(filepath, DataFrames.DataFrame)
end


# Scenario Generation Function, Monte Carlo
function simulate_prices(data)
    μ, α = 0, 9.05372307007077 # Mean and SD
    price = zeros(8760)
    for t in 1:8760
        ε = μ + α * randn()  # Generate a new random residual for each iteration
        price[t] =
          -10.5073 * data.S1[t] +  7.1459 * data.C1[t] +
            2.9938 * data.S2[t] +  3.6828 * data.C2[t] +
            0.6557 * data.S3[t] +  1.9710 * data.C3[t] + 
            0.9151 * data.S4[t] + -1.4882 * data.C4[t] + 
            0.0000705481036188987 * data.Load[t] +
            -0.000673131618513161 * data.VRE[t] +
            25.6773336136185 + ε
    end
    return price
end


# Scenario Plotting Function. Note: 8760 is the length of horizon
function plot_scenarios(data, n_plot)
    anim = Animation()
    first_scenario = simulate_prices(data)
    p = plot(1:8760, first_scenario, label="Scenario 1", xlim=(1, 8760), ylim=(minimum(first_scenario), maximum(first_scenario)), title="Price Scenarios Animation", xlabel="Hour", ylabel="Price [\$/MW]")
    for i in 2:n_plot
        new_scenario = simulate_prices(data)
        current_ylims = ylims(p)
        ylims!(p, (min(minimum(new_scenario), current_ylims[1]), max(maximum(new_scenario), current_ylims[2])))
        plot!(p, 1:8760, new_scenario, label="Scenario $i")
        frame(anim, p)
    end
    gif(anim, "SDDP_animation.gif", fps = 5)
end


# Static Price Plot Function
function static_price_plot(data, n_plot)
    Plots.plot([simulate_prices(data) for _ in 1:n_plot], legend = false, xlabel = "Hour", ylabel = "Price [\$/MW]")
    Plots.savefig("static_price_plot.png")
end


# Scenario Reduction Function Using K-Means Clustering
function reduce_scenarios(data, num_scenarios, k, iter)
    scenarios = [simulate_prices(data) for _ in 1:num_scenarios]
    scenarios = transpose(mapreduce(permutedims, vcat, scenarios))
    result = kmeans(scenarios, k; maxiter = iter, display = :iter)
    return result.centers, result  # Return reduced scenarios and the clustering result
end


# Reduced Scenarios Plotting and Histogram Function
function plot_reduced_scenarios(reduced_scenarios)
    histogram(reduced_scenarios, bins=30, label="Price Scenarios", legend=false, fmt=:png)
    xlabel!("Price")
    ylabel!("Frequency")
    title!("Price Scenario Distribution")
    Plots.savefig("reduced_scenario_histogram.png")
end


# Function to Create and Return Markovian Graph for SDDP Model
function create_sddp_markov_graph(reduced_scenarios, k, num_budget)
    global current_scenario_idx = 0

    # Internal function to call each column from reduced scenario matrix
    function cluster_simulator()
        num_reduced_scenarios = size(reduced_scenarios, 2)
        current_scenario_idx = mod(current_scenario_idx, num_reduced_scenarios) + 1
        return reduced_scenarios[:, current_scenario_idx]
    end

    # Create the Markovian Graph using the cluster simulator
    graph = SDDP.MarkovianGraph(cluster_simulator; budget = num_budget, scenarios = k)

    return graph
end


# Function to Define and Train SDDP Model
function define_and_train_sddp_model(graph, h, bound)
    # Define the SDDP Model
    Ptaker = SDDP.PolicyGraph(
        graph;
        sense = :Max,
        upper_bound = bound,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, node
        t, price = node
        E_min = 0
        p = 1_000 #MW
        E_max = h*p #MWh
        PF = 0
        eff_c = 0.725
        eff_dc = 0.507
        cost_op = 0
        rate_loss = 0
        ini_energy = E_max
        @variables(sp, begin
            0 <= e_stored <= E_max, SDDP.State, (initial_value = ini_energy)
            0 <= e_discharge <= p
            0 <= e_charge <= p
            E_min <= e_aval <= E_max
            e_loss
            z_charge, Bin
            z_discharge, Bin
        end)
        @expressions(sp, begin
            e_pur, e_charge / eff_c
            e_sold, e_discharge * eff_dc
        end)
        @constraints(sp, begin
            e_charge <= p * z_charge
            e_discharge <= p * z_discharge
            z_charge + z_discharge <= 1
            e_stored.out == e_stored.in + e_charge - e_discharge
            e_aval == (1 - rate_loss) * e_stored.out
        end)
        #Transiton Matrix and Constraints
        @constraints(
            sp, begin
            e_loss == (e_stored.out*rate_loss)
            e_stored.out == e_stored.in + e_charge - e_discharge
        end
        ) 
        # Define the stage objective
        SDDP.parameterize(sp, [(price,)]) do (ω,)
            cost_pur = @expression(sp, ω * e_pur)
            w_cost = @expression(sp, cost_pur + (cost_op * e_aval))
            w_rev = @expression(sp, ω * e_sold)
            @stageobjective(sp, w_rev - w_cost)
            return
        end
    end
    
        # Train the SDDP Model
    SDDP.train(
        Ptaker;
        time_limit = 10800.0,
        #stopping_rules = [SDDP.BoundStalling(10, 1e-4)],
    )
    return Ptaker
end

# Function to simulate the policy using closest nodes for out-of-sample testing
function simulate_with_closest_nodes(Ptaker, data, graph, n_replications)
    nodes = collect(keys(graph.nodes))
    # Function to find the closest node
    function closest_node(nodes, t, p)
        _, i = findmin([(t == t_ ? (p - p_)^2 : Inf) for (t_, p_) in nodes])
        return nodes[i]
    end
    # Price data in a vector named 'Price' from DataFrame
    p = (data[!, :Price])
    # Create a vector of stage-price pairs
    stage_price_pairs = [
        (closest_node(nodes, t, p[t]), p[t]) for 
        t in eachindex(p)
    ]
    # Create an out-of-sample historical sampling scheme
    out_of_sample = SDDP.Historical(stage_price_pairs)
    # Simulate the policy with the defined sampling scheme
    simulations = SDDP.simulate(
        Ptaker,
        n_replications,
        Symbol[:e_stored, :e_charge, :e_discharge, :z_charge, :z_discharge, :e_sold, :e_pur],
        sampling_scheme = out_of_sample,
    )
    println("Completed $(length(simulations)) simulations.")
    return simulations
end



# Function to Analyze and Plot Simulations
function analyze_and_plot_simulations(simulations, Ptaker, h_label)
    # Modify file paths to include h_label in the filenames
    results_filepath = "OoS_Historical_objective_values_and_confidence_interval_h$(h_label).txt"
    node_index_plot_path = "Phourly_node_index_h$(h_label).png"
    price_noise_plot_path = "Phourly_price_realizations_h$(h_label).png"
    spaghetti_plot_path = "PSDDP_spaghetti_plot_h$(h_label).html"
    SOC_result = "OoS_Historical_SOC_values_h$(h_label).txt"
    discharge_result = "OoS_Historical_e_discharge_values_h$(h_label).txt"
    charge_result = "OoS_Historical_e_charge_values_h$(h_label).txt"

    # Calculate Objective Values and Confidence Interval
    objectives = map(simulations) do simulation
        sum(stage[:stage_objective] for stage in simulation)
    end

    μ, ci = SDDP.confidence_interval(objectives)
    lower_bound = SDDP.calculate_bound(Ptaker)
    println("Confidence interval: ", μ, " ± ", ci)
    println("Lower bound: ", lower_bound)

    # Save Objective Values and Confidence Interval to a file
    open(results_filepath, "w") do file
        println(file, "Confidence interval: ", μ, " ± ", ci)
        println(file, "Lower bound: ", lower_bound)
    end


    State_of_Charge = map(simulations[1]) do node
        return node[:e_stored].out
    end
    # Convert the vector to a DataFrame
    soc_dataframe = DataFrame(State_of_Charge = State_of_Charge)

    # Save SOC data to a file
    CSV.write(SOC_result, soc_dataframe)


    #Save discharge energy Values
    discharge_energy = map(simulations[1]) do node
        return node[:e_discharge]
    end
    e_discharge_dataframe = DataFrame(discharge_energy = discharge_energy)
    CSV.write(discharge_result, e_discharge_dataframe)


    #Save charge energy Values
    charge_energy = map(simulations[1]) do node
        return node[:e_charge]
    end
    e_charge_dataframe = DataFrame(charge_energy = charge_energy)
    CSV.write(charge_result, e_charge_dataframe)    






    # Node Index Scenarios Plot
    Node_Index_realizations = Vector()
    for simulation in simulations
        node_indices = [stage[:node_index] for stage in simulation]
        push!(Node_Index_realizations, node_indices)
    end

    # Initialize a new plot for node indices
    node_index_plot = plot()  # This creates a new figure
    for (i, realization) in enumerate(Node_Index_realizations)
        plot!(node_index_plot, realization, label="Simulation $i", legend=true)
    end
    xlabel!(node_index_plot, "Stage")
    ylabel!(node_index_plot, "Node Index")
    title!(node_index_plot, "Node Index for All Simulations")
    savefig(node_index_plot, node_index_plot_path)

    # Noise Index Scenarios Plot
    price_realizations = Vector()
    for simulation in simulations
        noise_indices = [stage[:noise_term] for stage in simulation]
        push!(price_realizations, noise_indices)
    end

    # Initialize a new plot for price realizations
    price_noise_plot = plot()  # This creates a new figure
    for (i, realization) in enumerate(price_realizations)
        plot!(price_noise_plot, realization, label="Simulation $i", legend=true)
    end
    xlabel!(price_noise_plot, "Stage")
    ylabel!(price_noise_plot, "Price")
    title!(price_noise_plot, "Price Noise Realizations for All Simulations")
    savefig(price_noise_plot, price_noise_plot_path)

    # Spaghetti and Publication Plots
    chart = SDDP.SpaghettiPlot(simulations)
    SDDP.add_spaghetti(chart; title="Storage") do sim
        sim[:e_stored].out
    end
    SDDP.add_spaghetti(chart; title="Charge Energy") do sim
        sim[:e_charge]
    end
    SDDP.add_spaghetti(chart; title="Discharge Energy") do sim
        sim[:e_discharge]
    end
    SDDP.add_spaghetti(chart; title="Charge Decision") do sim
        sim[:z_charge]
    end
    SDDP.add_spaghetti(chart; title="Discharge Decision") do sim
        sim[:z_discharge]
    end
    SDDP.add_spaghetti(chart; title="Energy Sold") do sim
        sim[:e_sold]
    end
    SDDP.add_spaghetti(chart; title="Energy Bought") do sim
        sim[:e_pur]
    end
    SDDP.plot(chart, spaghetti_plot_path, open=false)

    # Additional Publication Plots
    for (title, symbol) in [("Storage", :e_stored), ("Charge Energy", :e_charge), ("Discharge Energy", :e_discharge), ("Charge Decision", :z_charge), ("Discharge Decision", :z_discharge),("Energy Sold", :e_sold,), ("Energy Bought", :e_pur)]
        SDDP.publication_plot(simulations; title=title) do sim
            if symbol == :e_stored
                return sim[symbol].out
            else
                return sim[symbol]
            end
        end
        Plots.savefig("PSDDP_$(replace(title, " " => "_"))_h$(h_label).png")
    end
end


#################################################################################################
# Main Script Execution
#################################################################################################
# Load data
data = load_data("SDDP_Hourly.csv")

# Set number of Scenarios
num_scenarios = 10000

# Set number of clusters
k = 1000

# Set maximum cluster iteration
Iter = 1000

# Set number of nodes for the policy graph
num_budget = 20000

# Set number of policy simulations
n_replications = 1  # Here We set number of simulations to 1 because we are only simulating one sequence of noise

# Scenario generation and plotting. We set number of plot scenarios to 10
plot_scenarios(data, 10)
static_price_plot(data, 10)

# Reduce scenarios
reduced_scenarios, result = reduce_scenarios(data, num_scenarios, k, Iter)

# Plot reduced scenarios and histogram
plot_reduced_scenarios(reduced_scenarios)

# Create Markov Graph
graph = create_sddp_markov_graph(reduced_scenarios, k, num_budget)

# Different scenarios for discharge duration 'h'
h_values = [24, 48, 168, 336, 720]

# Different Objective Bounds for different discharge durations
# NOTE: THIS WAS ESTIMATED BY MONTE CARLO SIMULATIONS OF STOCHASTIC SEQUENCE OF NOISE!!!!!!
# READ https://sddp.dev/stable/tutorial/warnings/#Choosing-an-initial-bound FOR MORE INFORMATION ON CHOOSING THE BOUND
bounds = [30_000_000, 35_000_000, 45_000_000, 60_000_000, 100_000_000]

# Run the model for different discharge durations with respective bounds
for (h, bound) in zip(h_values, bounds)
    Ptaker = define_and_train_sddp_model(graph, h, bound)
    simulations = simulate_with_closest_nodes(Ptaker, data, graph, n_replications)
    # Call the function with h_label parameter to distinguish files
    analyze_and_plot_simulations(simulations, Ptaker, h)
end
#################################################################################################