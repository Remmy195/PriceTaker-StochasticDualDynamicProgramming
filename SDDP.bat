@echo off
REM Run Monte Carlo simulation script
julia Monte_Carlo_Simulation.jl

REM Run SDDP script
julia Stochastic_Dual_Dynamic_Programming.jl

pause
