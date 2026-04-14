# NCCL Distributed 8-GPU Quarter-Degree Atmosphere Run — Notes

## Overview

Running a 1/4-degree global atmosphere simulation (moist baroclinic wave, DCMIP-2016)
on 8x NVIDIA H100 80GB GPUs using vanilla Oceananigans with NCCLDistributed architecture.

- **Branch**: `glw/microphysics-init` (Oceananigans pinned to `glw/nccl-distributed-solver`)
- **Date**: 2026-04-13
- **Machine**: Lightning.ai studio, 8x H100 80GB, CUDA driver 550.163.01 (CUDA 12.4)

## Configuration

| Parameter | Value |
|-----------|-------|
| Grid (global) | 1440 x 560 x 64 |
| Grid (per rank) | 360 x 280 x 64 |
| Partition | 4x2x1 (Rx=4, Ry=2) |
| Latitude | (-70, 70) |
| Longitude | (0, 360) |
| Column height | 30 km |
| Δt | 1.0 s |
| Halo | (4, 4, 4) |
| Advection | WENO(order=5), bounds-preserving for moisture |
| Microphysics | OneMomentCloudMicrophysics, NonEquilibriumCloudFormation (τ=30s) |
| Timestepper | SSPRK3 |
| Target sim time | ~1 year (30818 blocks x 1024 steps x 1s) |
| IC file | `quarter_deg_day1_cloud_tau30.jld2` |

## Scripts

- **`sharding/quarter_degree_nccl_distributed_run.jl`** — 8-GPU NCCL distributed (the main run)
- **`sharding/quarter_degree_vanilla_cuda_run.jl`** — single-GPU vanilla CUDA (for testing)
- **`sharding/quarter_degree_atmosphere_run.jl`** — Reactant-based distributed (alternative)

## Launch Command

```bash
~/.julia/bin/mpiexecjl -n 8 --project julia -O0 sharding/quarter_degree_nccl_distributed_run.jl 2>&1 | tee run_nccl_distributed.log
```

## Critical Fix: NCCL Library Compatibility

### Problem

NCCL_jll v2.28.3 (from Julia's artifact system) was built for CUDA 13.0, but:
- Julia's CUDA runtime is 12.9 (`LocalPreferences.toml`)
- The system CUDA driver is 550.163.01 (supports up to CUDA 12.4)

This caused `ncclCommInitRank` to **segfault** — NCCL 2.28.3 calls CUDA 13 APIs
that don't exist in the CUDA 12.x runtime, resulting in a null function pointer dereference.

### Failed Approaches

1. **Setting CUDA runtime to 13.0** (`CUDA.set_runtime_version!(v"13.0")`):
   CUDA 13 runtime requires driver >= 570, but we have driver 550 → "CUDA driver version
   is insufficient for CUDA runtime version"

2. **Using CUDA 12 NCCL_jll artifact**: Still had linking issues with `libcudart.so.12`

### Working Fix

PyTorch ships NCCL 2.27.3 that uses `dlopen` for CUDA libraries instead of hard-linking
against a specific CUDA version. Swap the NCCL_jll artifact:

```bash
# Backup original
NCCL_LIB=~/.julia/artifacts/8841a73f2e87b003d3d434a4fe21196aa3dd147b/lib
mv $NCCL_LIB/libnccl.so.2.28.3 $NCCL_LIB/libnccl.so.2.28.3.bak

# Copy PyTorch's NCCL (doesn't hard-link libcudart)
cp /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
   $NCCL_LIB/libnccl.so.2.27.3

# Update symlink
ln -sf libnccl.so.2.27.3 $NCCL_LIB/libnccl.so.2
```

**Verification**: `NCCL.Communicator(1, 0; unique_id=id)` succeeds after the swap.

> **Note**: This is a machine-specific workaround. On systems with CUDA 13-compatible drivers
> (>= 570), the default NCCL_jll artifact should work fine.

## CFL Constraint

The acoustic CFL limit for this grid:

```
Δz = H / Nz = 30000 / 64 ≈ 469 m
c_s ≈ 330 m/s (speed of sound)
Δt_max ≈ Δz / c_s ≈ 1.42 s
```

With SSPRK3 (3-stage), effective Δt_max is slightly larger but Δt=2s caused NaN after
~1025 steps. **Δt=1s is stable.**

## NCCLDistributed Architecture Setup

The NCCLDistributed type lives in the Oceananigans NCCL extension:

```julia
using MPI
MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
CUDA.device!(rank % length(CUDA.devices()))

const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributed = NCCLExt.NCCLDistributed

arch = NCCLDistributed(GPU(); partition = Partition(Rx, Ry, 1))
```

Key points:
- Each MPI rank must be assigned a unique GPU via `CUDA.device!`
- `mpiexecjl` (not system `mpiexec`) must be used since Julia's MPI.jl uses MPICH, not system PMIx
- Install via: `using MPI; MPI.install_mpiexecjl(; destdir=expanduser("~/.julia/bin"))`

## IC Loading (Distributed)

Each rank loads the full IC file, builds a GPU source grid, and interpolates to its local portion:

```julia
src_grid = LatitudeLongitudeGrid(GPU();
    size = (Nλ_src, Nφ_src, Nz_src),
    halo = halo,
    latitude = (-80, 80),     # source grid extent
    longitude = (-180, 180),
    z = (0, H))

# For each field:
src_field = Field(iloc, src_grid)
gpu_data = Oceananigans.on_architecture(GPU(), Array{FT}(src_array))
copyto!(Oceananigans.interior(src_field), gpu_data)
fill_halo_regions!(src_field)
interpolate!(target_field, src_field)
```

Important: source grid must be on GPU (not CPU) for `interpolate!` to work with GPU target fields.

## Performance (Preliminary)

- **Model build**: ~immediate
- **IC loading**: ~29s per rank (includes compilation)
- **First time step**: ~36s (72% compilation time)
- **Block 1 (1024 steps)**: pending...

## Known Issues / Gotchas

1. **Do NOT load Reactant on this machine** — it corrupts CUDA.jl's PTX target
2. Source IC grid uses longitude (-180, 180) while model uses (0, 360) — `interpolate!` handles this
3. Source IC grid latitude is (-80, 80), model is (-70, 70) — interpolation handles the subset
4. `any(isnan, parent(f))` works on GPU arrays for NaN checking
5. The `"This may cause errors"` warnings during precompilation are benign (CUDA extension warnings)
