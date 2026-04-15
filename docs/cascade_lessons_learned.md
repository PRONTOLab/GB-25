# Cascade lessons learned (1/8° → 1/16° → 1/24°)

Accumulated over the vanilla-Oceananigans + NCCLDistributed cascade on 8× H100 80 GB.
Captured 2026-04-15 after an 8-hour 1/24° run lost its final state to a save-check bug.

## Fatal mistakes (avoid at all cost)

### 1. Never check save times with float mod
```julia
# BAD — silently drops ~½ of saves
t > 0 && mod(t, output_interval_time) < Δt_max || return
```
`model.clock.time` accumulates `t += Δt` in Float64. After N steps the drift can land on
either side of the exact multiple of `output_interval_time`. Got lucky at iter 9000,
unlucky at iter 18000, unlucky at iter 27000 → 8h sim wall time, no final state saved.

**Fix:** iteration-based check — integers have no drift:
```julia
iter > 0 && mod(iter, save_iter_interval) == 0 || return
```
Pair with `IterationInterval(save_iter_interval)` at the callback level for belt+suspenders.

### 2. Never clamp moisture in a balanced evolved state
Setting negative `ρqᵛ / ρqᶜˡ / ρqᶜⁱ` to zero after loading from a file breaks
thermodynamic equilibrium. Microphysics then produces NaN tendencies at **any** Δt
(even 1e-6 s) when re-started. Only clamp on analytic cold-start ICs, never when
loading from a model-generated checkpoint.

### 3. Never assume a save succeeded — verify
At every expected save iteration, check `ls output_dir | grep iter<N>` AND
`grep "Saved rank" task_log`. If the callback silently misses, you want to know
within minutes, not at the end of an 8-hour run. If a miss happens, propose
kill+relaunch with a fix — don't just report and hope.

## Distributed-run specifics

### 4. Rank → tile mapping
For `NCCLDistributed(GPU(); partition=Partition(Rx, Ry, 1))`:
```julia
ix = rank ÷ Ry
iy = rank % Ry
```
Every derived script (assembly, direct-copy loader, viz) depends on this
convention. Get it wrong once and tiles end up transposed.

### 5. Face-field shapes on distributed Bounded grids
When saving `interior(field)` per-rank to JLD2:

| Field  | Topology   | iy=0 rank       | iy=Ry-1 (top) rank | z-extent |
|--------|------------|-----------------|--------------------|----------|
| ρ, ρθ, ρqᵛ, micros | Center×3      | (Nx, Ny, Nz)    | (Nx, Ny, Nz)       | Nz |
| ρu     | Face-x Periodic | (Nx, Ny, Nz) | (Nx, Ny, Nz)       | Nz |
| ρv     | Face-y Bounded  | (Nx, Ny, Nz) | **(Nx, Ny+1, Nz)** | Nz |
| ρw     | Face-z Bounded  | (Nx, Ny, **Nz+1**) | (Nx, Ny, **Nz+1**) | all |

Globally assembled shapes must preserve these:
- `ρv` → `(Nλ, Nφ+1, Nz)`, `ρw` → `(Nλ, Nφ, Nz+1)`, centers → `(Nλ, Nφ, Nz)`

Assembly code: use `has_extra_y` test on the tile sizes (see `assemble_checkpoint.jl`).

### 6. `load_ic!` with `interpolate!` doesn't scale to same-resolution restart
`load_ic!` creates a non-distributed `LatitudeLongitudeGrid(GPU(); …)` as the
source grid, so every GPU allocates the full global field. At 1/24° this is
8.5 GB/field × many fields → OOM.

**Fix for same-resolution restarts:** read only the rank's tile slice from the
assembled file and `copyto!` into `interior(target_field)` — no interpolation,
no source grid. See `load_ic_same_resolution!` in `twentyfourth_degree_continuation.jl`.

### 7. Oceananigans `JLD2Writer` is broken under `NCCLDistributed`
Grid metadata isn't serializable over NCCL. Workaround: custom per-rank JLD2
callback that writes `Array(interior(f))` for each field. Filename includes
`rank$r_iter$n` so downstream assembly can reconstruct.

### 8. MPI workers persist after rank-0 errors
If rank 0 aborts during init (e.g. OOM in `load_ic!`), the other 7 Julia
workers stay alive holding 80 GB GPU memory. The next relaunch then fails
with "Out of GPU memory trying to allocate 7.9 GiB" at `CUDA.device!()`.

**Fix:** after any failed launch, always check `nvidia-smi --query-gpu=memory.used`
and `kill -9` zombie Julia PIDs before relaunching.

## Stability & CFL (for 1/24° at Δt=0.8)

### 9. Vertical acoustic is the binding CFL constraint
With `Δz = H/Nz = 30000/64 = 469 m` uniform and `c_s ≈ 340 m/s`:
- Vertical acoustic:  `(c_s + |w|)·Δt/Δz ≈ 0.63` — **operating here**
- Horizontal acoustic at ±80°:  `(c_s + |u|)·Δt/Δx_min ≈ 0.39`
- Horizontal acoustic at equator:  `≈ 0.067`
- Pure advective (any direction):  `≈ 0.05`

Max stable Δt ≈ 1.3s. Going beyond needs thicker Δz or a vertically-implicit
acoustic solver.

### 10. Acoustic time step is ∼ microphysics time step
Microphysics coupling (cloud-formation rate 1/τ_cloud) is stiff. Practically,
Δt_max scales with τ_cloud. τ_cloud=30 required Δt ≤ ~0.05; τ_cloud=120 allows
Δt=0.8+. Don't reduce τ_cloud below 60 without reducing Δt proportionally.

## Resource scaling (8× H100 80 GB)

| Resolution | Nλ × Nφ × Nz | GB / field / rank | Wall / iter | OOM? |
|-----------:|:------------:|------------------:|------------:|:----:|
| 1/8°       | 2880 × 1280 × 64 | 0.11 GB      | ~0.3 s      | no   |
| 1/16°      | 5760 × 2560 × 64 | 0.47 GB      | ~0.6 s      | no   |
| 1/24°      | 8640 × 3840 × 64 | 1.06 GB      | ~1.1 s      | no (tight) |
| 1/32°      | 11520 × 5120 × 64 | 1.89 GB     | ~1.8 s est  | **YES** |

1/32° was OOM with ~50 model fields × 1.89 GB = 95 GB > 80 GB. Use 1/24° as
the practical ceiling on 8× H100 80 GB until sharding topology is widened
(16 GPUs) or model fields are reduced.

## File-size reference (1/24° at Nλ=8640, Nφ=3840, Nz=64, Float32)

- Per-rank save (`fields_rankN_iter*.jld2`): **19 GB** × 8 ranks = 152 GB
- Assembled single-file: **85 GB**
- 1h-cadence saves over 6h run: 6 × 85 GB = 510 GB assembled, 6 × 152 GB = 912 GB raw

## Cascade procedure (proven path)

1. **1/4° → 1/8°**: analytic IC, spinup a few hours on 4 GPUs
2. **1/8° → 1/16°**: cascade IC via `interpolate!`, 12 h on 8 GPUs at Δt=1.2
3. **1/16° → 1/24°**: cascade IC via `interpolate!`, 6–12 h on 8 GPUs at Δt=0.8
4. **1/24° restart**: direct-copy loader (same resolution), iter-based saves

Every transition: load assembled IC file, check all-finite + ρ-min > 0 before
first step, save early+often with iter-based cadence.
