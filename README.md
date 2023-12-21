#### Introduction
Companion code to Remi Akinwonmi's 2023 Master's Thesis "A stochastic dual dynamic programming approach to operational scheduling of long duration energy storage systems". This is a price taker model that incorporates stochastic elements to the price variable.

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
- `Deterministic_Optimal_Control.jl`: Implements deterministic control solutions with perfect foresight market scenarios.
- `Deterministic_Optimal_Control_RH.jl`: Rolling horizon strategy for deterministic optimal control.
- `Monte_Carlo_Simulation.jl`: Monte Carlo simulation of the deterministic optimal control model from a distribution.

NOTE:
- To run `Stochastic_Dual_Dynamic_Programming.jl`, set parameters and the distribution of random variable.
- The initial Bound on the objective value is estimated by running `Monte_Carlo_Simulation.jl` for a large number of scenarios and using the mean and confidence interval of the objective values to calibrate the Bound.
- `Stochastic_Dual_Dynamic_Programming.jl` performs In-Sample and Out-of-Sample evaluations of the SDDP policy.
- Set the type of policy evaluation.
- For out_of_sample_historic simulation, provide a vector of prices.

#### Documentation
- Detailed information can be found in the [SDDP.jl documentation](https://sddp.dev/stable).

#### License
- This project is under the [MIT License](LICENSE).

#### Contact
- [Remi Akinwonmi / reak4480@colorado.edu]
