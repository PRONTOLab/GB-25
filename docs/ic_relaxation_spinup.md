# IC-relaxation spinup strategy for high-resolution simulation

## Problem

To run a compressible atmosphere at resolution R (e.g. 1/16°), we'd like to
warm-start from a physically balanced state rather than burn wallclock on
a cold spinup. The natural source is an equilibrated lower-resolution
checkpoint (e.g. 1/8°) upsampled to R.

The challenge: linearly (or nearest-neighbor) interpolating a coarse field
onto a much finer grid introduces non-physical features on the target grid:

- Piecewise-constant or piecewise-bilinear "staircase" patterns at scales
  of the coarse cell size, invisible to the coarse simulation but resolved
  — and dynamically active — at the fine resolution.
- Sharp gradients at coarse-cell boundaries that drive spurious acoustic
  waves and feed moist-physics instabilities.
- Moisture and density fields slightly out of balance with the
  high-resolution dynamics, leading to fast-growing errors during the
  first minutes of simulation.

At 1/16° under compressible dynamics + WENO advection + bulk microphysics,
these artifacts have produced density NaNs within a few thousand time
steps when the IC was interpolated from a 1° balanced state.

## Strategy: relax to IC, then let go

Rather than try to smooth the IC ahead of time or use some spectral
filter, we add a **per-field linear relaxation** directly in the model
tendency:

```
    d(ρϕ)/dt += -α(t) * (ρϕ - ρϕ_IC)
```

for each prognostic ρϕ in {ρ, ρu, ρv, ρw, ρθ, ρqᵛ, optionally ρqᶜˡ, ρqᶜⁱ},
with a strength α(t) that decays linearly from an initial value α₀ at
t=0 down to zero over a spinup window T_decay, and stays at zero
thereafter:

```
    α(t) = max(0, α₀ · (1 − t / T_decay))
```

`ρϕ_IC` is a **persistent snapshot field** taken once, right after the IC
is loaded and interpolated onto the target grid. It is stored as an
extra Field alongside the prognostic state, and is never overwritten, so
the tendency sees a fixed target for the duration of the spinup.

### Why it works

During 0 ≤ t ≤ T_decay the relaxation term does two things:

1. **Dissipates the interpolation staircase.** Scales that are present in
   the target IC but not physically consistent at the target resolution
   generate a residual `ρϕ − ρϕ_IC` of order of the amplitude of those
   interpolation artifacts. The damping pulls the solution toward the
   same (imperfect but balanced-enough) target and bleeds energy out of
   the spurious modes rather than letting them cascade.

2. **Gives the moist physics time to nucleate cloud liquid / ice from
   the vapor field consistent with the upsampled density and
   temperature**, while the dynamics can only drift slowly away from IC.

By T_decay the relaxation turns off entirely. From that point on the
simulation evolves freely — no tendency modification, no ongoing
nudging, and in particular no reference to the (now stale) IC snapshot.

### How the snapshot is captured

Under Reactant/XLA we cannot run JLD2 I/O or `Oceananigans.interpolate!`
inside a compiled step. The snapshot must therefore be **built outside
the compile boundary** and then made available to the compiled tendency.

The current implementation allocates snapshot fields of the same layout
(and sharding) as each prognostic before `AtmosphereModel` is constructed,
wires them into a NamedTuple of `RelaxToICForcing{name}` objects, passes
that as `forcing = ...` to the model, loads the IC via the existing
`set_moist_baroclinic_wave_from_file!` path, and finally calls
`Oceananigans.set!(snapshot, prognostic)` once per field — which under
Reactant dispatches to a `@jit`-compiled elementwise copy kernel. The
snapshots are then captured by reference inside the forcing callables and
seen by the compiled time-step graph as additional read-only inputs.

### How the damping is scheduled

`α(t)` is a pure function of `clock.time`. It is evaluated per-cell in
the tendency kernel — cheaper than a field multiplication and trivially
traceable through XLA. Once `t > T_decay`, `α` is clamped to zero and
the forcing contributes nothing; the compiler cannot short-circuit the
multiply-add away, but the cost is negligible.

### Turning it off

The constructor takes `relaxation = nothing | (α0, T_decay)`. With the
default `nothing`, no snapshots are allocated, an empty forcing NamedTuple
is passed to `AtmosphereModel`, and the behavior is identical to the
pre-feature code. This is the knob to pull if we suspect the forcing of
interfering with a science diagnostic or scaling benchmark.

## Typical use

The 1/8° checkpoint used as the IC source is available at
<https://www.dropbox.com/scl/fi/rq206amnk6ylw0mz5gk1v/cascade_checkpoint.jld2?rlkey=v4rdkm7emjkib8s1k8e8352qk&dl=1>
(5.3 GB; drop-in for `simulations/initial_conditions/cascade_checkpoint.jld2`).

For a 1/16° spinup from this 1/8° checkpoint, with Δt = 0.5 s:

```julia
model = moist_baroclinic_wave_model(arch;
    Nλ = 5760, Nφ = 2560, Nz = 64,
    Δt = 0.5,
    initial_conditions_path = "…/cascade_checkpoint.jld2",
    interpolation_type = :linear,
    relaxation = (0.1, 1800.0),   # α0 = 0.1 s⁻¹, decay to 0 over 30 min
)
```

`α₀ = 0.1 s⁻¹` (equivalently, `τ₀ = 10 s`) is fast enough to squash
sub-grid noise within a handful of time steps but leaves larger-scale
dynamics mostly untouched. `T_decay = 30 min` gives the moist physics
time to equilibrate with the rebalanced fields. Both numbers are
tunable per-experiment.

## Caveats

- The snapshot is an **instantaneous IC**, not a slowly evolving target.
  This is intentional — we want the simulation free by T_decay — but it
  means the relaxation term does **not** substitute for a
  nudging-to-reanalysis run. It's a spinup aid only.
- Energy/momentum/mass are not conserved by the forcing (nothing in the
  form −α·(ρϕ−ρϕ_IC) guarantees integral constraints). For scientific
  analysis of conservation, restrict to post-T_decay output.
- The snapshots occupy additional device memory: 6 fields at the full
  prognostic layout (plus 2 more if `ρqᶜˡ`/`ρqᶜⁱ` are included), which
  for 1/16° on 16 GPUs is roughly 2 GiB per device. Check BFC limits
  before enabling on memory-tight configurations.
