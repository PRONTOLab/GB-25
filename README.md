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


### Oh the Woes of Running on Perlmutter

I blame HPE.

NCCL is giving grief. Here is a summary of what we've tried on PM.

1. If you use the "default" NCCL that comes with Reactant, then we see occasional (non-deterministic) segfaults.
2. Using the "official" NCCL modules on PM. The following will make sure that you're using the same NCCL as the NERSC nightly benchmarks (which work...)
    1. Add to jobscript: `ml load nccl/2.24.3`
    2. Add to `launcher.sh` the following preload: `export LD_PRELOAD=/global/common/software/nersc9/nccl/2.24.3/lib/libnccl.so`
  
Using the official NERSC NCCL module results in the following error:
* We see `cxil_map: write error` in the logs (that's why I suspect it's an HPE problem)
* We see the following NCCL error message in the `.err`:
```
E0000 00:00:1743742851.490303  583398 pjrt_stream_executor_client.cc:3072] Execution of replica 0 failed: INTERNAL: NCCL operation ncclGroupEnd() failed: unhandled system error (run with NCCL_DEBUG=INFO for details). Last NCCL warning(error) log entry (may be unrelated) '[Service thread] Error encountered progressing operation=Connect, res=3, closing connection'.
```
* We see the following logs when enabling `NCCL_DEBUG=INFO`
```
[...]
nid001264:1291155:1293014 [1] register_mr_buffers:647 NCCL WARN NET/OFI Unable to register memory (type = 2) for device 5. RC: -22, Error: Invalid argument
nid001264:1291155:1293014 [1] NCCL INFO transport/net.cc:829 -> 2
nid001264:1291155:1293040 [1] NCCL INFO transport.cc:194 -> 2
nid001264:1291155:1293040 [1] NCCL INFO group.cc:133 -> 2
nid001264:1291155:1293040 [1] NCCL INFO group.cc:75 -> 2 [Async thread]
nid001264:1291155:1291605 [1] NCCL INFO group.cc:423 -> 2
nid001264:1291155:1291605 [1] NCCL INFO group.cc:573 -> 2
nid001264:1291155:1291605 [1] NCCL INFO group.cc:106 -> 2

nid001264:1291155:1293015 [3] register_mr_buffers:647 NCCL WARN NET/OFI Unable to register memory (type = 2) for device 1. RC: -22, Error: Invalid argument
nid001264:1291155:1293015 [3] NCCL INFO transport/net.cc:829 -> 2
nid001261:1906368:1908393 [1] NCCL INFO [Proxy Progress] Device 1 CPU core 76
nid001261:1906368:1908394 [2] NCCL INFO [Proxy Progress] Device 2 CPU core 1
nid001261:1906368:1908389 [0] NCCL INFO Channel 01/1 : 4[0] -> 0[0] [receive] via NET/AWS Libfabric/7/GDRDMA/Shared
nid001261:1906368:1908387 [3] NCCL INFO Channel 01/1 : 7[3] -> 3[3] [receive] via NET/AWS Libfabric/1/GDRDMA/Shared
nid001261:1906368:1908390 [1] NCCL INFO Channel 01/1 : 5[1] -> 1[1] [receive] via NET/AWS Libfabric/5/GDRDMA/Shared
nid001261:1906368:1908389 [0] NCCL INFO Channel 05/1 : 4[0] -> 0[0] [receive] via NET/AWS Libfabric/7/GDRDMA/Shared
nid001261:1906368:1908387 [3] NCCL INFO Channel 05/1 : 7[3] -> 3[3] [receive] via NET/AWS Libfabric/1/GDRDMA/Shared
nid001261:1906368:1908390 [1] NCCL INFO Channel 05/1 : 5[1] -> 1[1] [receive] via NET/AWS Libfabric/5/GDRDMA/Shared
nid001261:1906368:1908388 [2] NCCL INFO Channel 01/1 : 6[2] -> 2[2] [receive] via NET/AWS Libfabric/3/GDRDMA/Shared
nid001261:1906368:1908388 [2] NCCL INFO Channel 05/1 : 6[2] -> 2[2] [receive] via NET/AWS Libfabric/3/GDRDMA/Shared

nid001261:1906368:1908364 [1] register_mr_buffers:647 NCCL WARN NET/OFI Unable to register memory (type = 2) for device 5. RC: -22, Error: Invalid argument
nid001261:1906368:1908364 [1] NCCL INFO transport/net.cc:973 -> 2
nid001261:1906368:1908390 [1] NCCL INFO transport/net.cc:434 -> 2
nid001261:1906368:1908390 [1] NCCL INFO transport.cc:213 -> 2
nid001261:1906368:1908390 [1] NCCL INFO group.cc:133 -> 2
nid001261:1906368:1908390 [1] NCCL INFO group.cc:75 -> 2 [Async thread]
nid001261:1906368:1906870 [1] NCCL INFO group.cc:423 -> 2

nid001261:1906368:1908364 [1] proxy.cc:1632 NCCL WARN [Service thread] Error encountered progressing operation=Connect, res=3, closing connection
nid001261:1906368:1906870 [1] NCCL INFO group.cc:573 -> 2
nid001261:1906368:1906870 [1] NCCL INFO group.cc:106 -> 2
[...]
```

Attempted workarounds:
1. Increase NCCL buffer size: `export NCCL_BUFFSIZE=33554432` (this is 8x the default)
2. Following the [NERSC recommendations](https://docs.nersc.gov/systems/perlmutter/vendorbugs/#nccl-hangs-on-perlmutter-in-nccl-tests-with-ofi-plugin-and-slingshot):
```
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
```
3. Add even more grasping at straws:
```
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export FI_MR_CACHE_MONITOR=kdreg2
export MPICH_GPU_SUPPORT_ENABLED=0
```

### Ludovic Fixed it!

Looks like this was missing: `export JULIA_CUDA_USE_COMPAT=false` 

* Current working verison on Perlmutter here: https://github.com/PRONTOLab/GB-25/tree/jpb/futzing-about-on-pm
