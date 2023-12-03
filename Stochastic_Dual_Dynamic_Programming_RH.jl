####################################################################################################################
####################################################################################################################
# Rolling Horizon Optimization of Long-Duration Energy Storage Price Taker Model Using Stochastic Dual Dynamic Model
####################################################################################################################
####################################################################################################################

# Load Packages
using CSV, DataFrames, Gurobi, Plots, SDDP, Base.Filesystem, BenchmarkTools
const GRB_ENV = Gurobi.Env() # This setup reuses a single Gurobi environment (GRB_ENV) for multiple model solves by passing the GRB_ENV object to Gurobi.Optimizer

# Function Definitions
# Data Loading Function
function load_data(filepath)
    CSV.read(filepath, DataFrame)
end

# Function to Simulate Prices from a Gaussian Distribution, Monte Carlo
function simulate_prices(data, μ, α, timesteps)
    price = zeros(length(timesteps))
    for (i, t) in enumerate(timesteps)
        ε = μ + α * randn()  # Generate a new random residual for each iteration
        price[i] = calculate_price(data, t, ε)
    end
    return price
end

function calculate_price(data, t, ε)
    -10.5073 * data.S1[t] + 7.1459 * data.C1[t] +
    2.9938 * data.S2[t] + 3.6828 * data.C2[t] +
    0.6557 * data.S3[t] + 1.9710 * data.C3[t] +
    0.9151 * data.S4[t] - 1.4882 * data.C4[t] +
    0.0000705481036188987 * data.Load[t] -
    0.000673131618513161 * data.VRE[t] +
    25.6773336136185 + ε
end

# Rolling Period Simulation Function
function rolling_period_simulation(data, horizon_length, opti_length, nodes, num_scenarios, h, p, bound)
    # Define constants
    μ, α = 0, 9.05372307007077  # Mean and SD
    total_hours = size(data, 1)
    start_hour = 1
    results_df = DataFrame(
    Time = Int[],
    SOC = Float64[],
    Charge = Float64[],
    Discharge = Float64[],
    Charge_decision = Float64[],
    Discharge_decision = Float64[],
    Energy_Bought = Float64[],
    Energy_Sold = Float64[],
    Initial = Float64[],
    )

    
    # Initialize initial state of charge
    initial_soc = h*p
    
    while start_hour <= total_hours - horizon_length + 1
        # Define horizon range
        horizon_range = start_hour:(min((start_hour + horizon_length), total_hours))
        
        prices = simulate_prices(data, μ, α, horizon_range)

        Monte_Carlo() = prices

        # Create the Markovian Graph using the cluster simulator
        graph = SDDP.MarkovianGraph(Monte_Carlo; budget = nodes, scenarios = num_scenarios)

        # Create and train the SDDP model
        Ptaker = create_and_train_sddp_model(graph, horizon_range, initial_soc, h, p, bound)
        
        # Run simulation and update results
        initial_soc = run_simulation_and_update_results(Ptaker, results_df, start_hour, opti_length, horizon_range, data, graph)
        
        # Update the start hour for the next horizon window
        start_hour += opti_length
    end
    
    return results_df
end

# Function to create and train the SDDP model for a given horizon
function create_and_train_sddp_model(graph, horizon_range, initial_soc, h, p, bound)
    #Model
    Ptaker = SDDP.PolicyGraph(
        graph;
        sense = :Max,
        upper_bound = bound,
        optimizer = () -> Gurobi.Optimizer(GRB_ENV),
    ) do sp, node
        t, price = node
        E_min = 0
        E_max = h*p #MWh
        PF = 0
        eff_c = 0.725
        eff_dc = 0.507
        cost_op = 0
        rate_loss = 0
        #initial_soc = E_max
        @variables(sp, begin
            0 <= e_stored <= E_max, SDDP.State, (initial_value = initial_soc)
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

    # Train the sub-model
    SDDP.train(
        Ptaker;
        iteration_limit = 50,
        #time_limit = 20.0,
        log_file = "SDDP_RH.log",
        #stopping_rules = [SDDP.BoundStalling(10, 1e-4)],
    )
    return Ptaker
end

# Function to run the simulation for the trained model and update the results DataFrame
# NOTE: We perform out of sample simlation of the optimal policy with a deterministic noise
function run_simulation_and_update_results(Ptaker, results_df, start_hour, opti_length, horizon_range, data, graph)
    # Price data in a vector named 'Price' from DataFrame
    p = [data[t, :Price] for t in horizon_range]
    nodes = collect(keys(graph.nodes))
    # Function to find the closest node
    function closest_node(nodes, t, p)
        _, i = findmin([(t == t_ ? (p - p_)^2 : Inf) for (t_, p_) in nodes])
        return nodes[i]
    end
    # Create a vector of stage-price pairs
    stage_price_pairs = [
        (closest_node(nodes, t, p[t]), p[t]) for 
        t in eachindex(p)
    ]
    # Create an out-of-sample historical sampling scheme
    out_of_sample = SDDP.Historical(stage_price_pairs)
        
    # Simulate the sub-model
    n_replications = 1
    simulations = SDDP.simulate(
        Ptaker,
        n_replications,
        Symbol[:e_stored, :e_charge, :e_discharge, :z_charge, :z_discharge, :e_pur, :e_sold],
        sampling_scheme = out_of_sample,
    )

    # After simulation, extract the data
    SOC_sim = [sim[:e_stored].out for sim in simulations[1]]
    initial_soc = Int(round(SOC_sim[opti_length]))
    e_charge_values = [sim[:e_charge] for sim in simulations[1]]
    e_discharge_values = [sim[:e_discharge] for sim in simulations[1]]
    charge_decision_values = [sim[:z_charge] for sim in simulations[1]]
    discharge_decision_values = [sim[:z_discharge] for sim in simulations[1]]
    e_bought_values = [sim[:e_pur] for sim in simulations[1]]
    e_sold_values = [sim[:e_sold] for sim in simulations[1]]

    # Append the results to the DataFrame
    for t in eachindex(SOC_sim)
        push!(results_df, (
            Time = start_hour + t - 1,
            SOC = SOC_sim[t],
            Charge = e_charge_values[t],
            Discharge = e_discharge_values[t],
            Charge_decision = charge_decision_values[t],
            Discharge_decision = discharge_decision_values[t],
            Energy_Bought = e_bought_values[t],
            Energy_Sold = e_sold_values[t],
            Initial = initial_soc
        ))
    end
    
    println(length(SOC_sim));

    # Extract the value of e_stored.out at the end of the 24-hour period
    # This will be the initial state for the next horizon
    #global initial = Int(round([sim[:e_stored].out for sim in simulations[1]][look_ahead_length]))
    return initial_soc
end

######################################################################################################################################
# Main Script Execution
######################################################################################################################################
# Load data
data = load_data("SDDP_Hourly.csv")

# Number of Scenarios
num_scenarios = 1000

#Number of nodes for the policy graph
num_budget = [100, 200, 300, 800, 2000, 3000]

# Define parameters for the rolling horizon
horizon_length_values = [23, 47, 71, 191, 359, 743] # Rolling periods
opti_length = 24

# Different scenarios for discharge duration 'h'
h_values = [24, 48, 168, 336, 720]

# Power Output
p = 1_000 #MW

# Different Estimated Objective Bounds for different discharge durations
# NOTE: THIS WAS ESTIMATED BY MONTE CARLO SIMULATIONS OF STOCHASTIC SEQUENCE OF NOISE!!!!!!
# READ https://sddp.dev/stable/tutorial/warnings/#Choosing-an-initial-bound FOR MORE INFORMATION ON CHOOSING THE BOUND
bounds = [40_000_000, 45_000_000, 50_000_000, 60_000_000, 70_000_000]

# Run the model for different discharge durations and rolling periods
for (h, bound) in zip(h_values, bounds)
    for (horizon_length, nodes) in zip(horizon_length_values, num_budget)
        results_df = rolling_period_simulation(data, horizon_length, opti_length, nodes, num_scenarios, h, p, bound)
        # Save the results to a CSV file
        CSV.write("SDDP_RH($horizon_length)_results_h$h.csv", results_df)    
    end
end
######################################################################################################################################