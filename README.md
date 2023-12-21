#### Introduction
Companion code to Remi Akinwonmi's 2023 Master's Thesis "A stochastic dual dynamic programming approach to operational scheduling of long duration energy storage systems". This is a price taker model that incorporates stochastic elements to the price variable.


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
- `Stochastic_Dual_Dynamic_Programming.jl`: Implements stochastic dual dynamic programming for optimal decision-making considering electeicity price uncertainity.
- `Stochastic_Dual_Dynamic_Programming_RH.jl`: Focuses on rolling horizon implementations in stochastic dual dynamic programming, enhancing short-term decision accuracy.
- `SDDP_OoS_Simulator.jl`: Conducts out-of-sample testing of strategies, crucial for understanding model performance in unseen market scenarios.
- `SDDP_OoS_Historic_Simulation.jl`: Conducts out-of-sample testing of strategies using historical or forecasted price data. This can be used to compare model performance with deterministic optimal control with perfect foresight.
- `Deterministic_Optimal_Control.jl`: Offers deterministic control solutions for perfect foresight market scenarios.
- `Deterministic_Optimal_Control_RH.jl`: Rolling horizon approach in deterministic optimal control

#### Documentation
- Detailed information can be found in the [SDDP.jl documentation](https://sddp.dev/stable).

#### License
- This project is under the [MIT License](LICENSE).

#### Contact
- [Remi Akinwonmi / reak4480@colorado.edu]
