# READ THIS FIRST:

This is a special branch of the `GB-25` repo that is linked to the DJ4Earth differentiable ESM repository: [https://github.com/DJ4Earth/differentiable-esm-components-2025](https://github.com/DJ4Earth/differentiable-esm-components-2025). This is released in combination with the manuscript submission "DJ4Earth: Differentiable, and Performance-portable Earth System Modeling via Program Transformations"

To replicate the numerical examples in the paper featuring Oceananigans:

1. Instantiate the environment with the given Project.toml and Manifest.toml files.
2. Run `julia -O0 --project correctness/abernathy_channel.jl` to run the model with AD.
3. Run `julia -O0 --project correctness/makie_abernathy.jl` to produce plots from the produced data.

To change the directory where model data and graphs are produced, edit lines 30 in `abernathy_channel.jl` and 17 in `makie_abernathy.jl`.
You can also change the number of timesteps in the model spinup and AD run in lines 254 and 284, respectively, of `abernathy_channel.jl` (Reactant requires the number of iterations to be hardcoded).

### The following README is part of the original GB-25 Repository

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

## Running scaling benchmarks

Initial setup:

* If you haven't already, install Julia following the instructions on the [official website](https://julialang.org/downloads/).
  Julia may (or may not!) be available on the system you use, check that first too
* Enter the directory and precompile the environment with
  ```
  julia --project -e 'using Pkg; Pkg.instantiate()'
  ```
  This step will take a few minutes, but should be needed only the first time (or any time you want to update the package)

We have some scripts which use sharding in the [`sharding/`](./sharding) directory.
If you have multiple devices available locally, you may be able to launch a simple problem locally.
To do this, you first must comment out `Reactant.Distributed.initialize()` in `sharding/simple_sharding_problem.jl`.
Next, enter `sharding/` and type
```
julia --project -O0 simple_sharding_problem.jl
```
(If you do not have multiple devices availalbe and you are using a CPU, add the flag `XLA_FLAGS="--xla_force_host_platform_device_count=4"` to
trick XLA into thinking that you have 4 individual devices available, for example.)
On systems we tested this application (Alps @ CSCS, Leonardo @ CINECA, Perlmutter @ NERSC) we provide some script to automatically submit scaling jobs that you can run with (need to be again inside the sharding directory, and in this case you do ***not*** need to comment out `Reactant.Distributed.initialize()`)
```
julia alps_scaling_test.jl simple_sharding_problem.jl
julia leonardo_scaling_test.jl simple_sharding_problem.jl
julia perlmutter_scaling_test.jl simple_sharding_problem.jl
```
You'll need to tweak the content of the scaling test scripts to what you need, e.g. change `submit` to `true` to actuallt run the jobs, `time` to the walltime requested for the job, `Ngpus` for the number of GPUs you want to run on (note that that's number of GPUs , not nodes!), etc., have a look at the script for your system.
