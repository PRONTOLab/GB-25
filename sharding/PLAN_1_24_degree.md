# 1/24° NCCL-distributed spinup plan

## Machine
8x NVIDIA H100 80GB, Lightning.ai studio, vanilla Oceananigans + NCCLDistributed (no Reactant).

## Grid
- Nλ=8640, Nφ=3840, Nz=64, lat=(-80, 80), lon=(0, 360), H=30 km
- Partition: Rx=4, Ry=2 → per-rank 2160 × 1920 × 64 ≈ 265 M cells
- Halo: (4, 4, 4)

## IC source
`checkpoint_step_008193.jld2` — 1/8° (Nλ=2880, Nφ=1280, Nz=64, lat=(-80, 80))
with cloud fields: ρ, ρu, ρv, ρw, ρθ, ρqᵛ, micro_ρqᶜˡ, micro_ρqᶜⁱ, ρqʳ, ρqˢ.

## Working configuration
| Parameter | Value |
|-----------|-------|
| Δt | 0.5 s |
| τ_cloud (NonEquilibriumCloudFormation) | 120 s |
| SST anomaly | +2 K |
| IC-relaxation | off |
| Cloud-condensate damping | off |
| ρqᵛ / ρqᶜˡ / ρqᶜⁱ / ρqʳ / ρqˢ clamp to ≥ 0 after interpolation | **yes** |
| Microphysics | OneMomentCloudMicrophysics, mixed-phase (liquid + ice) |
| Advection | WENO(order=5), bounds-preserving [0,1] for moisture/cloud/precip |
| Timestepper | SSPRK3 |

## Stability test result
Δt=0.5 ran 500 iters (250 sim seconds) across all 8 ranks with no NaN.
Δt=1.0 produced NaN at the first time step.

## CFL context
- Acoustic CFL: Δz/c_s = 469/330 = 1.42 s
- Vertical (|w|+c_s)/Δz at CFL=0.5: Δt ≤ 0.64 s
- Horizontal at pole (|u|+c_s)/dx_80 at CFL=0.5: Δt ≤ 0.77 s
- Δt=0.5 sits within the CFL=0.5 envelope for all directions.

## Key findings from experiments
1. **Cloud-field IC is essential.** Without pre-populated cloud liquid/ice, the first microphysics step triggers a cold-start condensation burst that drives latent heat → buoyancy → vertical velocity amplification → NaN. With cloud fields present, this loop is absent.

2. **Moisture clamping required.** Source IC has small negative ρqᵛ, ρqᶜˡ, ρqᶜⁱ, ρqʳ values (~-1e-4) from bounds-preserving WENO. These negatives, amplified by interpolation + microphysics, produce NaN in specific tiles. Clamping all moisture/condensate fields to ≥ 0 after `interpolate!` removes that source.

3. **τ_cloud controls the Δt ceiling (without cloud IC).** With no cloud IC:
   - τ=30 → Δt < 0.02 (even with IC-relaxation)
   - τ=120 → Δt ≤ 0.05 (with IC-relaxation)
   - τ=2400 → Δt ≤ 0.05 (without relaxation, with clamp)
   With cloud IC: τ=120 allows Δt=0.5 with no relaxation.

4. **IC-relaxation forcing works and helps** (but isn't needed with cloud IC).
   - Requires `Adapt.adapt_structure` on `RelaxToICForcing` for vanilla GPU kernels.
   - Requires `fill_halo_regions!` on snapshot fields after copy.
   - ρ forcing is a no-op: Breeze's `_compute_density_tendency!` ignores `forcing.ρ`.

5. **Cloud-condensate damping (ρqᶜˡ, ρqᶜⁱ → 0) doesn't help** because the feedback loop goes through latent heat on ρθ, not through cloud mass accumulation.

## Planned run
- Δt=0.5, 3 sim hours = 21600 iterations
- Per-iter diagnostic: throttle to every 100 iters (reduce wall overhead from ~1.1s/iter to ~0.5s/iter)
- Output writer: disabled (JLD2 serialization issue with distributed grid)
- Estimated wall time: ~3 hours

## Launch command
```bash
~/.julia/bin/mpiexecjl -n 8 --project julia -O0 sharding/twentyfourth_degree_nccl_distributed_run.jl 2>&1 | tee run_24th_deg.log
```
