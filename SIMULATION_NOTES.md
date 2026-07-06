# Running atmosphere/ocean simulations on this node

Hard-won notes from getting vanilla-Oceananigans GPU atmosphere runs
to launch successfully on this Lightning H200 box.

## Hardware

- **GPU:** 1× NVIDIA H200, 144 GiB HBM, sm_90 (Hopper).
- **Driver:** 570.211.01 (production branch, ships with CUDA 12.8 support; CUDA.jl forward-compat reaches 13.x via artifacts).
- **System CUDA toolkit:** `/usr/local/cuda` is **CUDA 13.0** (`nvcc --version`).
- **CUDA.jl artifact runtime:** **13.2** (chosen automatically).

A 1/8° (2880×1280×64) moist-baroclinic-wave model with full microphysics
sits comfortably in ~65 GiB out of 144 GiB. Plenty of headroom.

## Three things that will bite you (and how to fix them)

### 1. `LD_LIBRARY_PATH` leaks system CUDA into Julia

The login shell ships with:
```
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
```

That last entry pulls system CUDA 13.0 `.so`s (libcusparse, libcublas, …)
into the Julia process *instead of* CUDA.jl's bundled 13.2 artifacts.
Mixing the two yields:

```
CUDA error: device kernel image is invalid (code 200, ERROR_INVALID_IMAGE)
```

…on the very first GPU kernel launch — typically `gpu_compute_Δx_Az!`
inside `LatitudeLongitudeGrid` precomputed-metrics. CUDA.jl prints a
warning early that you should **not** ignore:

> CUDA runtime library `libcusparse.so.12` was loaded from a system path

**Fix:** strip the cuda toolkit lib path from `LD_LIBRARY_PATH` for every
Julia launch. Keep the nvidia driver libs.

```bash
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 julia ...
```

Sanity-check with `using CUDA; CUDA.versioninfo()`. If you do **not** see
the "loaded from a system path" warning and CUBLAS reads as 13.x from
artifacts, you're clean.

### 2. Reactant in the same process poisons Oceananigans GPU kernels

This is the nasty one. Even if you scrub `LD_LIBRARY_PATH`, **just
loading `Reactant`** in the Julia process — without ever calling it —
breaks every subsequent Oceananigans GPU kernel JIT with the same
`ERROR_INVALID_IMAGE`. A trivial KernelAbstractions kernel still works,
and a bare `using Oceananigans, CUDA` driver works, but the moment
`Reactant` enters the picture the next `LatitudeLongitudeGrid(GPU(); …)`
construction dies.

Reactant transitively replaces (or shadows) something CUDA.jl needs for
PTX codegen / linking. I did not chase the exact mechanism — the workaround
is simply: **do not mix Reactant and Oceananigans GPU in the same process.**

This means **`using GordonBell25` is poisonous too** for vanilla-CUDA runs,
because GB25's source files do `using Reactant` at the top, and Julia
package loading is eager.

**Fix:** for vanilla-Oceananigans GPU runs, write the script to use only
`Oceananigans + Breeze + CUDA + JLD2 + KernelAbstractions`. If you need
helpers from GB25 (`moist_baroclinic_wave_model`, `set_moist_baroclinic_wave_from_file!`,
the IC math, …), inline the bits you need rather than importing the
package.

For Reactant runs, the inverse rule applies: `using Reactant` is fine
because you're driving the GPU through XLA, not through CUDA.jl directly.

### 3. `using Breeze` does not load microphysics

`Breeze` keeps its CloudMicrophysics integration in a package extension
that only activates when `CloudMicrophysics` is in the loading process:

```julia
using Breeze
using CloudMicrophysics  # ← REQUIRED to activate BreezeCloudMicrophysicsExt

ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
ext.OneMomentCloudMicrophysics(; cloud_formation = ...)
```

Without that line, `Base.get_extension` returns `nothing` and the next
field access blows up with `type Nothing has no field OneMomentCloudMicrophysics`.

GB25's `src/moist_baroclinic_wave_model.jl` has `using CloudMicrophysics`
near the top — when you fork the model construction into your own script,
copy that import too.

## Time stepping API surface

Oceananigans's `time_step!` requires an explicit `Δt`:

```julia
Oceananigans.TimeSteppers.time_step!(model, Δt)
```

`GordonBell25` wraps this in `time_step!(model)` (single arg) that reads
`Δt = model.clock.last_Δt + 0`. If you bypass GB25, mirror that pattern
yourself or pass `Δt` explicitly. For `AtmosphereModel`'s first step,
also call `Oceananigans.TimeSteppers.update_state!(model)` first
(see `src/timestepping_utils.jl:44`).

## Δt stability for moist baroclinic wave

Empirical stability limits for 1/8° (2880×N×64) WENO(5) + SSP-RK3
with full OneMomentCloudMicrophysics (cloud_τ=30s):

| Config             | Δt=0.5 | Δt=1.0 | Δt=1.5 | Δt=2.0 |
|--------------------|--------|--------|--------|--------|
| ±80° (Nφ=1280)    | ✅     | ✅     | —      | ❌     |
| ±70° (Nφ=1120)    | ✅     | ✅     | ❌     | ❌     |

The binding constraint is the **vertical acoustic CFL**:

```
Δt < Δz / c_s ≈ (30 km / 64) / 340 m/s ≈ 1.38 s
```

independent of horizontal resolution. SSP-RK3 extends this somewhat
but the effective stability bound with full physics is between 1.0 and 1.5.
**Δt=1.0 is the safe, proven choice.** Don't go higher.

Other instability sources found:
- **cloud_τ=100** (too slow relaxation): NaN within 1 block at Δt=1
- **WENO(3) + Δt=1.0**: NaN (order reduction)
- **Known stable combo**: WENO(5), Δt=1.0, cloud_τ=30, halo=(8,8,8)

## Throughput on H200

1/8° (2880×N×64), vanilla CUDA, Δt=1.0, WENO(5):

| Config         | s/step | s/block (512 steps) | SYPD     |
|----------------|--------|---------------------|----------|
| ±80° (1280)    | 1.04   | 534                 | 0.00001  |
| ±70° (1120)    | 0.89   | 455                 | 0.00001  |

## Multi-resolution simulation pipeline

Proven pipeline on this node (all vanilla CUDA, single H200):

1. **1° dry spinup** (360×160×64) — 14 sim days — preexisting IC
2. **1/4° moist** (1440×640×64, Δt=1, cloud_τ=30) — 1 sim day — ~6.5h wall
3. **1/8° ±80°** (2880×1280×64) from 1/4° checkpoint — 6 sim hours — ~6.2h wall
4. **1/8° ±70°** (2880×1120×64) from 1/8° ±80° checkpoint — 6 sim hours — ~5.4h wall

Each stage uses `Oceananigans.Fields.interpolate!` for cross-resolution IC loading.
Source grid must be on the **same architecture** as the target (GPU→GPU); CPU→GPU fails.

## IC interpolation gotcha

When loading ICs from a checkpoint at a different resolution or latitude range,
the source grid in `load_ic_from_file!` must be constructed on the **same GPU
architecture** as the model grid. Using `CPU()` for the source will fail:

```
"Cannot interpolate! because from_field is on CPU() while to_field is on GPU"
```

Also use `set!(src_field, FT.(src_array))` to upload host data to device,
not `.=` assignment.

## Initial condition files

JLD2 checkpoint layout:

| Key  | Shape           | Notes |
|------|-----------------|-------|
| ρ    | Nz × Nφ × Nλ    | CenterField (HDF5 stores reversed) |
| ρu   | Nz × Nφ × Nλ    | XFaceField (periodic-λ) |
| ρv   | Nz × (Nφ+1) × Nλ | YFaceField |
| ρw   | (Nz+1) × Nφ × Nλ | ZFaceField |
| ρθ   | Nz × Nφ × Nλ    | CenterField (potential temperature density) |
| ρqᵛ  | Nz × Nφ × Nλ    | CenterField (water vapor density) |
| Nλ, Nφ, Nz | Int        | grid dims |
| time, iteration, last_Δt | scalars | clock state |

Note: cloud fields (ρqᶜˡ, ρqᶜⁱ, ρqʳ, ρqˢ) are **not** saved by the current
`save_state!` — only the 6 fields above. If you need cloud diagnostics,
either add them to `save_state!` or derive RH from θ/ρ/qᵥ.

## Recommended launch recipe — vanilla Oceananigans GPU

```bash
cd ~/atmos/GB-25   # or ~/ocean/GB-25
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  julia -O0 --project=. simulations/your_script.jl \
  > simulations/logs/run_$(date -u +%Y-%m-%dT%H-%M-%S).log 2>&1 &
```

Notes:
- **`-O0` is fine for GPU runs** — it doesn't affect GPU kernel codegen,
  only Julia-side compilation time (faster startup).
- Drop the cuda toolkit path from `LD_LIBRARY_PATH` (#1).
- Use a script that doesn't `using Reactant` or `using GordonBell25` (#2).
- For long runs, log to a file and tail it; the run is silent inside
  `time_step!` between `@time`/`@info` boundaries — pinned GPU util at
  100% with no new log lines is normal, **not** a hang.
- Add `flush(stdout); flush(stderr)` after every `@info` — stderr is
  block-buffered by default and will delay log output by minutes.

## Recommended launch recipe — Reactant compile runs

```bash
cd ~/atmos/GB-25   # or ~/ocean/GB-25
julia -O0 --project=. sharding/sharded_atmosphere_simulation_run.jl \
  --grid-x N --grid-y N --grid-z N
```

- `-O0` per the existing convention (Reactant compile runs).
- `LD_LIBRARY_PATH` cleanup is **probably** not needed because Reactant
  brings its own CUDA stack — but it doesn't hurt.
- These runs use `ReactantState()` arch and a separate code path; the
  `gpu_compute_Δx_Az!` issue does not appear.
- Keep `Reactant.jl` pinned to `main`.

## How to monitor a long run

```bash
# Liveness
ps -p $PID -o pid,stat,etime,%cpu,%mem

# GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader

# Log tail
tail -f simulations/logs/<logfile>.log
```

Healthy state during time-stepping:
- proc `R`, ~100% CPU on a single thread
- GPU util 80–100%, memory ~52-65 GiB for 1/8°
- log silent (only `@info`/`@time` boundaries flush)
