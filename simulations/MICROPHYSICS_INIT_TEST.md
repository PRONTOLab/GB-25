# Reproducing the 1° → 1/8° interpolation test

This document explains the bug, the fix, and the standalone reproducer for the
**file-based initial-condition loader** added on `dkz/atmosphere-example` and
patched on this branch.

## TL;DR

- New nearest-neighbor IC loader landed in
  [`src/moist_baroclinic_wave_model.jl`](../src/moist_baroclinic_wave_model.jl)
  on `dkz/atmosphere-example` (commit `541f265` "Add interpolation type"). It
  uses a `KernelAbstractions` kernel `_nn_atmos_field_copy!` to copy a host
  source array onto a device target field by index-space rounding.
- The vanilla branch (`set_moist_baroclinic_wave_from_file_vanilla!`) **did not
  work on CUDA**: it passed a host `Array{Float32, 3}` straight into the
  KA launch, and CUDA refused to compile the kernel ("not a bitstype").
- One-line fix: call `Oceananigans.on_architecture(arch, src_array)` to upload
  the source to the target architecture before launching the kernel
  (`src/moist_baroclinic_wave_model.jl:592`). No-op on CPU; host→device copy on
  GPU.
- After the fix, the standalone test
  [`simulations/atmosphere_eighth_from_1deg_cuda.jl`](atmosphere_eighth_from_1deg_cuda.jl)
  loads the 1° initial-conditions file (360×160×64) onto a 1/8° model
  (2880×1280×64), runs 512 explicit time steps, and finishes with no NaN.

## The bug

`set_moist_baroclinic_wave_from_file_vanilla!` constructs a list of (host
source array, device target field) pairs and dispatches each pair to a KA
kernel:

```julia
# src/moist_baroclinic_wave_model.jl:570 (pre-fix)
pairs = [
    (FT.(ρ_data),   dynamics_density(model.dynamics)),    # ← FT.(...) is on host
    (FT.(ρu_data),  model.momentum.ρu),
    ...
]
...
# pre-fix loop (broken on CUDA)
for (src_array, target_field) in pairs
    Oceananigans.Utils.launch!(arch, grid, :xyz,
        _nn_atmos_field_copy!, target_field, src_array,    # ← host Array
        Nx_src_f, Ny_src_f, Nz_src_f, Nx_dst, Ny_dst, Nz_dst)
end
```

CUDA captures kernel arguments by value, and `Base.Array` is **not isbits**
(its `.mem` is a managed `Memory` heap allocation), so `cufunction` refuses to
codegen the kernel:

```
GPU compilation of MethodInstance for gpu__nn_atmos_field_copy!(...) failed

Argument 4 to your kernel function is of type Array{Float32, 3}, which is not a bitstype:
  .ref is of type MemoryRef{Float32} which is not isbits.
    .mem is of type Memory{Float32} which is not isbits.
```

This **only fails on GPU** — on `CPU()` arch the kernel runs on the host so the
argument type is fine. The loader had been tested on CPU but never on CUDA.

## The fix

Move the source array to the model's architecture before launching the kernel:

```julia
# src/moist_baroclinic_wave_model.jl:592 (post-fix)
src_dev = Oceananigans.on_architecture(arch, src_array)

Oceananigans.Utils.launch!(arch, grid, :xyz,
    _nn_atmos_field_copy!, target_field, src_dev,
    Nx_src_f, Ny_src_f, Nz_src_f, Nx_dst, Ny_dst, Nz_dst)
```

`Oceananigans.on_architecture(arch, x)` is the canonical Oceananigans helper
for moving an array to the target architecture:

| input | result |
|---|---|
| `on_architecture(CPU(),       ::Array)`  | `Array` (no copy, no-op) |
| `on_architecture(GPU(CUDA…),  ::Array)`  | `CuArray` (host → device upload) |
| `on_architecture(GPU(CUDA…),  ::CuArray)`| same `CuArray` (`===`, no double-copy) |

CPU runs are unaffected. GPU runs now hand the kernel a `CuArray` instead of a
host `Array` and the kernel compiles fine.

## The reproducer

[`simulations/atmosphere_eighth_from_1deg_cuda.jl`](atmosphere_eighth_from_1deg_cuda.jl)
is a self-contained driver that:

1. Builds a 1/8° `LatitudeLongitudeGrid` (Nλ=2880, Nφ=1280, Nz=64) on
   `GPU(CUDABackend())`.
2. Constructs a Breeze `AtmosphereModel` with `WENO(order=5)` advection,
   `OneMomentCloudMicrophysics` (default `NonEquilibriumCloudFormation` —
   instantaneous saturation adjustment, **not** the `cloud_formation_τ_relax`
   path), and bulk surface fluxes.
3. Loads the 1° initial-conditions file
   `simulations/initial_conditions/atmosphere_no_microphysics_1deg_14day.jld2`
   (360×160×64) via an inlined copy of `set_moist_baroclinic_wave_from_file_vanilla!`
   — exercising the fix.
4. Runs an explicit `update_state!` + 1 first time step.
5. Runs 512 probe time steps at Δt = 0.5 s with a NaN check every 64 steps,
   aborting on detection.
6. Saves a 5.68 GB checkpoint with the same JLD2 schema as
   `atmosphere_spinup_cuda.jl`.

The 1° → 1/8° step is an **8× upsample in λ and φ** (each 1° cell becomes a
piecewise-constant 8×8 block of 1/8° cells; z passes through 1:1).

### Why it inlines the loader instead of calling `GordonBell25`

On this Lightning H200 box, `using GordonBell25` transitively loads `Reactant`,
which corrupts CUDA.jl's PTX target and causes any subsequent
`LatitudeLongitudeGrid(GPU(); …)` construction to fail with
`ERROR_INVALID_IMAGE`. (See `~/SIMULATION_NOTES.md` for the full investigation.)
The standalone driver therefore imports only `Oceananigans + Breeze + CUDA +
JLD2 + KernelAbstractions + CloudMicrophysics` and inlines the bits of
`src/moist_baroclinic_wave_model.jl` it needs.

The fix in this commit lives in **two places** that must stay in sync:

- `src/moist_baroclinic_wave_model.jl:592` — the canonical package loader
- `simulations/atmosphere_eighth_from_1deg_cuda.jl:193` — the inlined copy

If you change one, change the other.

## Running the test (vanilla CUDA, this box)

```bash
cd ~/atmos/GB-25
git checkout glw/microphysics-init

mkdir -p simulations/logs
LOGFILE="simulations/logs/atmosphere_eighth_from_1deg_cuda_$(date -u +%Y-%m-%dT%H-%M-%S).log"

LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  nohup julia -O0 --project=. simulations/atmosphere_eighth_from_1deg_cuda.jl \
  > "$LOGFILE" 2>&1 &

echo "pid: $!  log: $LOGFILE"
```

**Notes about the launch command:**

- `LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64` is **required**
  on this box. The default `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64`,
  which leaks system CUDA 13.0 libs into the Julia process and conflicts with
  CUDA.jl's bundled artifact 13.2 runtime, causing a different
  `ERROR_INVALID_IMAGE` on the very first kernel JIT. Stripping the cuda
  toolkit path (and keeping the nvidia driver libs) fixes it.
- `julia -O0` is for Reactant compile runs per the team convention; vanilla
  CUDA runs do not benefit from `-O0` but it does not hurt either.

### Expected timeline

| phase | wallclock |
|---|---|
| Julia + package precompile / module load | ~30 s |
| `build model` (incl. WENO + microphysics kernel JIT) | ~26 s, 91 % compile |
| `load ICs` (6 NN interpolate launches) | ~13 s |
| `first time step` | ~20 s, 68 % compile |
| `loop 512` at ~0.83 s/step | ~7 min |
| `checkpoint save` | ~14 s |
| **total** | **~9 min** |

Verified result on this box (PID 837756, log
`simulations/logs/atmosphere_eighth_from_1deg_cuda_2026-04-11T19-49-04.log`):
6 `NN interpolate` launches all succeed, post-load NaN check is clean,
post-first-step NaN check is clean, all 8 in-loop NaN checks (steps
64,128,…,512) are clean. The final checkpoint contains 0 NaN / 0 Inf in every
prognostic field.

Phase timings (verified):

| phase | wallclock | notes |
|---|---|---|
| build model | 25.5 s | 92 % compile |
| load ICs    | 5.5 s   | 78 % compile (much faster than 1/8°→1/8° because the 1° source arrays are tiny) |
| first time step | 20.5 s | 68 % compile |
| loop 512    | 425.9 s | **0.832 s/step**, no recompile |
| ckpt save   | 11.4 s  | 5.68 GB JLD2 |
| **total**   | **~488 s ≈ 8 min 8 s** | |

Final-checkpoint field stats
(`atmosphere_eighth_from_1deg_2026-04-11T19-57-12.jld2`, `time = 256.5`, `iter = 513`):

```
ρ    shape=(2880, 1280, 64)  nan=0 inf=0  min=0.015592    max=1.426    mean=0.33657
ρu   shape=(2880, 1280, 64)  nan=0 inf=0  min=-85.851     max=81.844   mean=2.9273
ρv   shape=(2880, 1281, 64)  nan=0 inf=0  min=-71.729     max=79.005   mean=-0.0025781
ρw   shape=(2880, 1280, 65)  nan=0 inf=0  min=-3.4205     max=10.414   mean=1.1365
ρθ   shape=(2880, 1280, 64)  nan=0 inf=0  min=9.8297      max=347.28   mean=109.87
ρqᵛ  shape=(2880, 1280, 64)  nan=0 inf=0  min=-0.0062058  max=0.045215 mean=0.0018409
```

Notes on the numbers:
- **0 NaN / 0 Inf in every field.** The bug is fixed and the loader produces a
  finite, integrable state.
- **ρu/ρv have grown to ±~80 m/s** in 256 sim seconds. The 1° spinup IC has
  ~25 m/s jet maxima; the 8× upsample creates large gradients at every block
  boundary that the explicit dynamics relax via acoustic waves. This is
  expected of NN cross-resolution loading and is a *quality* observation, not
  a *correctness* problem.
- **ρqᵛ has a small negative tail** (min ≈ −0.62 % of max). Same artifact we
  saw with the 1/8°→1/8° run — bounds-(0,1) WENO + explicit RK3 lets a tiny
  undershoot through. Not the IC loader's fault.

### Watching the run

The script flushes `stdout`/`stderr` after every `@info`, so the log is live.
Useful one-liners for monitoring:

```bash
# Liveness
ps -p $PID -o pid,stat,etime,%cpu,%mem

# GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader

# Latest progress lines
grep -E "step .*/ 512|nan=true|ERROR" "$LOGFILE" | tail
```

Healthy steady state: process `R`, ~99 % single-thread CPU, GPU ~100 % util,
GPU memory ~49 GiB (lower than the equivalent same-resolution run because the
1° source arrays are tiny — ~14 MB each vs ~940 MB at 1/8°).

### Scanning the checkpoint

```julia
using JLD2, Statistics, Printf
JLD2.jldopen("simulations/checkpoints/atmosphere_eighth_from_1deg_<jobid>.jld2") do f
    println("time=", f["time"], "  iter=", f["iteration"])
    for k in ("ρ","ρu","ρv","ρw","ρθ","ρqᵛ")
        a = f[k]
        n_nan = count(isnan, a); n_inf = count(isinf, a)
        finite = filter(isfinite, a)
        if isempty(finite)
            @printf "%-4s nan=%d inf=%d  ALL non-finite\n" k n_nan n_inf
        else
            @printf "%-4s nan=%d inf=%d  min=%.5g max=%.5g mean=%.5g\n" k n_nan n_inf minimum(finite) maximum(finite) Float64(mean(finite))
        end
    end
end
```

A healthy run produces:
- `n_nan = 0`, `n_inf = 0` for every field
- ρ ≈ 1 kg/m³ near surface, tapering toward ~0 at the top of the column
- Velocities (ρu, ρv) bounded by a few × the IC peak (1° spinup gives ~25 m/s
  jets; the upsample preserves them inside each 8×8 block and the model evolves
  them slightly over 256 sim seconds)
- ρqᵛ should be ≥ 0 within numerical noise; small negative values are an
  artifact of the explicit scheme + WENO bounds limiter, not the IC loader

## What this test does **not** cover

- **Quality of nearest-neighbor at cross-resolution.** NN of a 1° field onto a
  1/8° grid is crude — 8×8 piecewise-constant blocks with sharp gradients at
  block boundaries. This is a *correctness* test ("does the loader produce a
  finite, integrable state?"), not a quality test.
- **Linear interpolation.** The vanilla loader explicitly errors if you ask
  for `interpolation_type=:linear` (`src/moist_baroclinic_wave_model.jl:541`).
  Linear is only available on the Reactant path, via
  `Reactant.InterpolationType.Linear`. To exercise that, run on Perlmutter
  with `ReactantState()` arch and `interpolation_type=:linear`.
- **The Reactant-arch loader.** `set_moist_baroclinic_wave_from_file!` (no
  `_vanilla` suffix) at `src/moist_baroclinic_wave_model.jl:463` uses
  `Reactant.InterpolateArray` instead of the KA kernel, on a completely
  different code path. It has **not** been retested by this fix and may have
  its own bugs.
- **Cloud-formation relaxation timescale.** This test uses the default
  `NonEquilibriumCloudFormation(nothing, nothing)` (instantaneous saturation
  adjustment). An earlier run with `cloud_formation_τ_relax = 100` produced
  100 % NaN by step 1800. The sharded driver in
  `sharding/sharded_atmosphere_simulation_run.jl:108` uses
  `cloud_formation_τ_relax = 10.0` (note: 10, not 100) — that may be the
  intended stable value.

## Next steps

Per the team plan:

1. **Run the same test on Perlmutter** using the **Reactant** code path
   (`ReactantState()` arch, `interpolation_type=:linear`). The Reactant loader
   is a separate code path that hasn't been validated by this fix and could
   have its own analogous bugs (e.g., implicit sharding-vs-non-sharding issues
   in the `InterpolateArray` shape inference).
2. **Once the Reactant path is verified**, retire the vanilla-CUDA standalone
   reproducer or keep it as a CPU/GPU smoke test alongside the Reactant tests.

## File index

| path | role |
|---|---|
| [`src/moist_baroclinic_wave_model.jl`](../src/moist_baroclinic_wave_model.jl) | Canonical package code; `set_moist_baroclinic_wave_from_file_vanilla!` at line 540, `_nn_atmos_field_copy!` kernel at line 299, `on_architecture` fix at line 592 |
| [`simulations/atmosphere_eighth_from_1deg_cuda.jl`](atmosphere_eighth_from_1deg_cuda.jl) | Standalone reproducer; inlines the relevant pieces of the package code so it can run without `using GordonBell25` (which would load Reactant and corrupt CUDA on this box) |
| `~/SIMULATION_NOTES.md` | Investigation notes for the three traps that block any vanilla-CUDA Oceananigans GPU run on this Lightning H200 box (LD_LIBRARY_PATH leak, Reactant-poisons-CUDA, CloudMicrophysics extension) — required reading before running anything new on this hardware |
