# GB-25

This repository accompanies a submission for the 2025 Gordon Bell climate prize submission that showcases Reactant-acceleration of Oceananigans, ClimaOcean, and SpeedyWeather simulations.

## Package organization

### `oceananigans-dynamical-core`

* `super_simple_simulation.jl`: a barebones Oceananigans simulation that illustrates Reactant-acceleration of an Oceananigans simulation (but doesn't do anything interesting), useful for debugging.

* `baroclinic_wave_test.jl`: implements a still-simple Reactant-accelerated Oceananigans simulation of "baroclinic instability" (the physical process that generates weather in Earth's midlatitudes), which invokes more physics than "super simple simulation", and has the potential to generate pretty movies.

* `bumpy_baroclinic_wave_test.jl`: the same as "baroclinic wave test", but with some bathymetry. This test will eventually encapsulate all of the Oceananigans functionality needed for ocean climate simulations with ClimaOcean.

