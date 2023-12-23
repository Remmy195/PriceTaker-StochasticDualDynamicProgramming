#### Introduction
Companion code to Remi Akinwonmi's 2023 Master's Thesis "A stochastic dual dynamic programming approach to operational scheduling of long duration energy storage systems." This is a price-taker model that incorporates stochastic elements into the price variable.

READ [SDDP.jl documentation](https://sddp.dev/stable) BEFORE RUNNING THE SCRIPTS!

#### Getting Started
1. **Prerequisites**: 
   - Julia must be installed on your system. [Download Julia](https://julialang.org/downloads/).
   - To run this script, ensure the following packages are installed in your Julia environment:
      - CSV
      - DataFrames
      - Gurobi
      - Plots
      - SDDP
      - Clustering
      - StatsPlots
      - Distributions

You can install these packages by running the following commands in your Julia environment:

```julia
using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Gurobi")  # Note: Gurobi is a commercial solver and may require a license.
Pkg.add("Plots")
Pkg.add("SDDP")
Pkg.add("Clustering")
Pkg.add("StatsPlots")
Pkg.add("Distributions")
```

#### Scripts Overview
- `Stochastic_Dual_Dynamic_Programming.jl`: Implements stochastic dual dynamic programming for optimal decision-making considering electricity price uncertainty.
- `Stochastic_Dual_Dynamic_Programming_RH.jl`: Implements a rolling horizon strategy with various levels of foresight.
- `Monte_Carlo_Simulation.jl`: Monte Carlo simulation of the deterministic optimal control model from a distribution.
- `SDDP.bat`: This batch file runs both the Monte Carlo simulation and the SDDP script sequentially.
- `SDDP.sh`: This shell script runs both the Monte Carlo simulation and the SDDP script sequentially.

NOTE:
- To run `Stochastic_Dual_Dynamic_Programming.jl`, set parameters and the distribution of random variable.
- The initial Bound on the objective value is estimated by running `Monte_Carlo_Simulation.jl` for a large number of scenarios and using the mean and confidence interval of the objective value to calibrate the Bound.
- `Stochastic_Dual_Dynamic_Programming.jl` performs In-Sample and Out-of-Sample evaluations of the SDDP policy.
- Set the type of policy evaluation.
- For out_of_sample_historic simulation, provide a vector of prices.
- All simulation scripts draw data from `SDDP_Hourly.csv`

#### Running the Batch Script
- On Windows, run the `SDDP.bat` file
- On Linux/Mac, run the `SDDP.sh` file
   - This will execute the Monte Carlo simulation first, which calculates and saves the bounds (The bound is stochastic). Then, it runs the SDDP script, which reads these bounds and proceeds with its computation.
- ##### Manual Computation of Bounds
   If you prefer to set the bounds manually instead of having them computed by the Monte Carlo simulation, you can do so by modifying the SDDP script:
   - Open the SDDP script (`Stochastic_Dual_Dynamic_Programming.jl`) in your IDE.
   - Find the section where the script reads the bounds from the file and comment out this section to prevent the script from reading the file like below.
     ```julia
     # Read the bounds from the file
     #bounds_file = "calculated_bounds.txt"
     #bounds = Float64[]
     #if isfile(bounds_file)
     #    for line in readlines(bounds_file)
     #        push!(bounds, parse(Float64, line))
     #    end
     #end
     ```
  - Uncomment the line where bounds are set manually. Adjust the values as deemed fit.
    ```julia
    bounds = [35_000_000, 40_000_000, 50_000_000, 50_000_000, 70_000_000]
    ```
  - Save the changes and run only the SDDP script.

    
#### Documentation
- Detailed information can be found in the [SDDP.jl documentation](https://sddp.dev/stable).

#### License
- This project is under the [MIT License](LICENSE).

#### Contact
- [Remi Akinwonmi / reak4480@colorado.edu]
