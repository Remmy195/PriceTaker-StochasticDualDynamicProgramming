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
    gif(anim, "SDDP_animation.gif", fps = 3)
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
function define_and_train_sddp_model(graph, h, data, bound)
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
        SDDP.parameterize(sp, [(price, nothing)]) do (ω, _)
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
        sampling_scheme = SDDP.SimulatorSamplingScheme(() -> simulate_prices(data)),
        time_limit = 10800.0,
        #stopping_rules = [SDDP.BoundStalling(10, 1e-4)],
    )
    return Ptaker
end


# Simulation and Data Extraction Function
# We perfomed out of sample simulations of the optimal policy by generating new scenarios with the Monte Carlo scenario generator
function simulate_and_extract_data(Ptaker, n_replications, data)
    simulations = SDDP.simulate(
        Ptaker,
        n_replications,
        Symbol[:e_stored, :e_charge, :e_discharge, :z_charge, :z_discharge, :e_sold, :e_pur];
        sampling_scheme = SDDP.SimulatorSamplingScheme(() -> simulate_prices(data))
    )
    println("Completed $(length(simulations)) simulations.")
    return simulations
end


# Function to Analyze and Plot Simulations
function analyze_and_plot_simulations(simulations, Ptaker, h_label)
    # Modify file paths to include h_label in the filenames
    results_filepath = "OoS_Simulator_objective_values_and_confidence_interval_h$(h_label).txt"
    node_index_plot_path = "hourly_node_index_h$(h_label).png"
    spaghetti_plot_path = "SDDP_spaghetti_plot_h$(h_label).html"
    
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
    for (title, symbol) in [("Storage", :e_stored), ("Charge Energy", :e_charge), ("Discharge Energy", :e_discharge), ("Charge Decision", :z_charge), ("Discharge Decision", :z_discharge), ("Energy Sold", :e_sold), ("Energy Bought", :e_pur)]
        SDDP.publication_plot(simulations; title=title) do sim
            if symbol == :e_stored
                return sim[symbol].out
            else
                return sim[symbol]
            end
        end
        Plots.savefig("SDDP_$(replace(title, " " => "_"))_h$(h_label).png")
    end
end


#################################################################################################
# Main Script Execution
#################################################################################################
# Load data
data = load_data("SDDP_Hourly.csv")

# Number of Scenarios
num_scenarios = 10000

# Number of clusters
k = 1000

# Set maximum cluster iteration
Iter = 1000

#Number of nodes for the policy graph
num_budget = 20000

# Set number of policy simulations
n_replications = 10

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
bounds = [35_000_000, 40_000_000, 50_000_000, 60_000_000, 100_000_000]

# Run the model for different discharge durations with respective bounds
for (h, bound) in zip(h_values, bounds)
    Ptaker = define_and_train_sddp_model(graph, h, data, bound)
    simulations = simulate_and_extract_data(Ptaker, n_replications, data)
    # Call the function with h_label parameter to distinguish files
    analyze_and_plot_simulations(simulations, Ptaker, h)
end
#################################################################################################