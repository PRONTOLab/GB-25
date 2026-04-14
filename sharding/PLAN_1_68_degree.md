# 1/68° NCCL-distributed run plan

## Machine
64x GH200 GPUs on CSCS Alps (Clariden), 16 nodes × 4 GPUs/node.
Vanilla Oceananigans + NCCLDistributed (no Reactant).

## Grid
- Nλ=24480, Nφ=10880, Nz=64, lat=(-80, 80), lon=(0, 360), H=30 km
- Δλ = Δφ = 1/68° ≈ 1.64 km at the equator
- Partition: Rx=8, Ry=8 → per-rank 3060 × 1360 × 64 ≈ 266 M cells
- Halo: (4, 4, 4)
- Total cells: 17.0 billion

## Resolution context
| | 1/24° (previous) | 1/68° (this plan) | 1/112° (true 1 km) |
|---|---|---|---|
| Δx at equator | 4.64 km | **1.64 km** | 0.99 km |
| Δx at 80° (zonal) | 805 m | **284 m** | 172 m |
| Nλ × Nφ | 8640 × 3840 | **24480 × 10880** | 40320 × 17920 |
| Total cells | 2.12 B | **17.0 B** | 46.2 B |
| GPUs needed | 8 | **64** | 256 |

## IC source
`checkpoint_step_008193.jld2` — 1/8° (Nλ=2880, Nφ=1280, Nz=64, lat=(-80, 80))
with cloud fields: ρ, ρu, ρv, ρw, ρθ, ρqᵛ, micro_ρqᶜˡ, micro_ρqᶜⁱ, ρqʳ, ρqˢ.

Fallback: `atmosphere_coarsened_1536x768x64.jld2` (1536×768×64, downloadable via
`simulations/download_atmosphere_ic_artifact.jl`). Lower resolution source but
has the same cloud fields needed for stable startup.

## Working configuration (inherited from 1/24° plan)
| Parameter | Value |
|-----------|-------|
| Δt | 0.2 s |
| τ_cloud (ConstantRateCondensateFormation) | 120 s |
| SST anomaly | +2 K |
| IC-relaxation | off |
| Cloud-condensate damping | off |
| ρqᵛ / ρqᶜˡ / ρqᶜⁱ / ρqʳ / ρqˢ clamp to ≥ 0 after interpolation | **yes** |
| Microphysics | OneMomentCloudMicrophysics, mixed-phase (liquid + ice) |
| Advection | WENO(order=5), bounds-preserving [0,1] for moisture/cloud/precip |
| Timestepper | SSPRK3 |

## CFL analysis
Scaling from 1/24° (where Δt=0.5 s was stable and Δt=1.0 produced NaN at step 1):

| Constraint | 1/24° | 1/68° |
|---|---|---|
| Δx at 80° (zonal, binding) | 805 m | 284 m |
| Horizontal CFL=0.5 limit (\|u\|+c_s)/Δx | Δt ≤ 0.77 s | Δt ≤ 0.27 s |
| Acoustic CFL Δz/c_s (Δz=469 m) | Δt ≤ 1.42 s | Δt ≤ 1.42 s |
| Vertical CFL=0.5 (\|w\|+c_s)/Δz | Δt ≤ 0.64 s | Δt ≤ 0.64 s |
| **Safe Δt** | **0.5 s** | **0.2 s** |

Vertical CFL is unchanged (same Nz=64, H=30 km). The horizontal CFL at 80°
latitude dominates: the 1/68° zonal spacing is 805/284 ≈ 2.83× tighter.

## Memory budget
- Per-rank cells: 3060 × 1360 × 64 = 266.1 M (matches 1/24° density of 265 M/rank)
- Estimated memory per GPU: ~75 GB (fits GH200 96 GB / H100 80 GB)

## Planned run
- Δt=0.2 s, 3 sim hours = 54,000 iterations
- Per-iter diagnostic: throttle to every 100 iters
- Output writer: disabled (JLD2 serialization issue with distributed grid)
- Estimated per-iteration time: ~0.5 s/iter (similar cells/rank to 1/24°)
- **Estimated wall time: ~7.5 hours**

## Data download
```bash
# Download the coarsened IC (1536×768×64, ~5.4 GB) if checkpoint_step_008193.jld2 is not available:
julia --project -O0 simulations/download_atmosphere_ic_artifact.jl atmosphere_coarsened_1536x768x64.jld2
```

## Launch command
```bash
cd /users/gwagner/atmos/GB-25
sbatch sharding/submit_sixtyeighth_degree_16node.sh
```
