# GB-25

This repository accompanies a submission for the 2025 Gordon Bell climate prize submission that showcases Reactant-acceleration of Oceananigans, ClimaOcean, and SpeedyWeather simulations.

## Package organization

* `src` implements two models:
    * `GordonBell25.data_free_climate_simulation_init`, which uses ClimaOcean to drive an Oceananigans `HydrostaticFreeSurfaceModel`
    * `GordonBell25.baroclinic_instability_model`, a simpler, unforced, pure Oceananigans setup.
    * Both models can be to use either a `LatitudeLongitudeGrid` or a `TripolarGrid` with idealized bathymetry.

* `simulations`: scripts to i) compile and ii) run either of the two models

* `sharding`: scripts and utilities that use XLA's sharding to distribute computations across multiple nodes. Oceananigans uses its `Distributed` architecture to represent sharding across nodes.

* `ext`: many small packages that each precompile part of a model time-step, in order to accelerate compilation during intensive jobs.
