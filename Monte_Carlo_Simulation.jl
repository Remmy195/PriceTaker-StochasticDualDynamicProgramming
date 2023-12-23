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

    objective_val = objective_value(Ptaker)

    return objective_val
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
data = CSV.read("SDDP_Hourly.csv", DataFrames.DataFrame)

# Distribution method
method_type = :normal_distribution  # or :t_distribution for student t distribution

# Discharge duration scenarios
h_values = [24, 48, 168, 336, 720]
num_iterations = 1000 #can be adjusted

# Initialize summary DataFrame
summary_info = DataFrame(h = Int[], Mean_Objective = Float64[], CI_Lower = Float64[], CI_Upper = Float64[])

for h in h_values
    objective_vals = Float64[]

    for i in 1:num_iterations
        price = simulate_prices(data, method_type)
        objective_val = optimize_price_deterministic(data, h, price)
        push!(objective_vals, objective_val)
        println("Objective Value for h=$(h): ", objective_val)
    end

    # Calculate mean and confidence interval
    mean_objective = mean(objective_vals)
    std_dev = std(objective_vals)
    confidence_level = 0.95
    df = num_iterations - 1
    t_value = quantile(TDist(df), 1 - (1 - confidence_level)/2)
    margin_error = t_value * std_dev / sqrt(num_iterations)
    confidence_interval = (mean_objective - margin_error, mean_objective + margin_error)

    push!(summary_info, (h, mean_objective, confidence_interval[1], confidence_interval[2]))
end

# Write summary to a CSV file
csv_filename = "Monte_Carlo_Results.csv"
CSV.write(csv_filename, summary_info)

# Display the mean profit and CI for all discharge durations
for row in eachrow(summary_info)
    println("Discharge Duration: $(row.h) hours")
    println("Mean Objective Value: $(row.Mean_Objective)")
    println("95% Confidence Interval: ($(row.CI_Lower), $(row.CI_Upper))")
end

# Safety margin for SDDP Bound. This is a percentahe and can be adjusted.
# We set this to have an optimistic bound that doesn't cut off a feasible solution
safety_margin_percentage = 20

# Calculate the initial bound for SDDP
for row in eachrow(summary_info)
    upper_confidence_bound = row.CI_Upper
    adjusted_bound = upper_confidence_bound * (1 + safety_margin_percentage / 100)
    println("Initial Bound for SDDP with Discharge Duration $(row.h) hours: ", adjusted_bound)
end
