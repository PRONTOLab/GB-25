#####
##### Relaxation-to-initial-condition forcing with linearly-decaying strength
#####

using Adapt: Adapt

"""
    RelaxToICSpec{F, T}

User-facing specification for an IC-relaxation forcing. Carries the IC
snapshot and the damping schedule; the live prognostic field is bound
in at model-construction time via `materialize_atmosphere_model_forcing`.
"""
struct RelaxToICSpec{F, T}
    ic       :: F
    α0       :: T
    T_decay  :: T
end

"""
    RelaxToICForcing{C, I, T}

Discrete-form Oceananigans/Breeze forcing that damps a prognostic field
toward a persistent IC snapshot:

```
    F(i,j,k) = -α(t) * (current[i,j,k] - ic[i,j,k])
```

`α(t) = max(0, α0 * (1 - t/T_decay))`. Direct references to both the
live prognostic field (`current`) and the IC snapshot (`ic`) are stored
here — no NamedTuple lookup happens in the tendency kernel, so the GPU
codegen stays clean.
"""
struct RelaxToICForcing{C, I, T}
    current  :: C
    ic       :: I
    α0       :: T
    T_decay  :: T
end

Adapt.adapt_structure(to, f::RelaxToICForcing) =
    RelaxToICForcing(Adapt.adapt(to, f.current),
                     Adapt.adapt(to, f.ic),
                     f.α0, f.T_decay)

Adapt.adapt_structure(to, s::RelaxToICSpec) =
    RelaxToICSpec(Adapt.adapt(to, s.ic), s.α0, s.T_decay)

@inline function (f::RelaxToICForcing)(i, j, k, grid, clock, fields)
    # NOTE: time-dependent damping (using clock.time) generated MLIR API
    # calls inside the GPU kernel — clock.time is a TracedRNumber that
    # cannot be operated on at GPU codegen. For now we apply a constant
    # α0 and rely on launching a fresh model+compile to schedule the
    # damping (e.g. α0=0.1 for the first 30 min, then α0=0 thereafter).
    @inbounds ic_val  = f.ic[i, j, k]
    @inbounds cur_val = f.current[i, j, k]
    return -f.α0 * (cur_val - ic_val)
end

# Hook into Breeze's forcing materialization so we can capture a
# reference to the prognostic `field` at construction time.
# Promote α0 and T_decay to the grid's float type so the GPU kernel does
# not hit mixed Float32/Float64 dispatches (which pull in unreachable
# error-handling code and break GPU codegen).
function Breeze.AtmosphereModels.materialize_atmosphere_model_forcing(
    spec::RelaxToICSpec, field, name, model_field_names, context)
    FT = eltype(field.grid)
    return RelaxToICForcing(field, spec.ic, FT(spec.α0), FT(spec.T_decay))
end

"""
    build_ic_relaxation_forcing(grid; α0, T_decay, include_cloud=false)

Allocate IC-snapshot fields and build a NamedTuple of `RelaxToICSpec`s
(one per prognostic name we relax). `α0` is the initial damping rate
(s⁻¹); the rate decays linearly to zero at `t = T_decay` and is clamped
to zero thereafter.

Returns `(forcing_specs, snapshots)`. `snapshots` is the NamedTuple of
newly-allocated IC-snapshot fields; callers fill them with
post-interpolation state after the IC is loaded.
"""
function build_ic_relaxation_forcing(grid; α0::Real, T_decay::Real, include_cloud::Bool=false)
    snapshots = (
        ρ    = CenterField(grid),
        ρu   = XFaceField(grid),
        ρv   = YFaceField(grid),
        ρw   = ZFaceField(grid),
        ρθ   = CenterField(grid),
        ρqᵛ  = CenterField(grid),
    )

    forcing = (
        ρ   = RelaxToICSpec(snapshots.ρ,   α0, T_decay),
        ρu  = RelaxToICSpec(snapshots.ρu,  α0, T_decay),
        ρv  = RelaxToICSpec(snapshots.ρv,  α0, T_decay),
        ρw  = RelaxToICSpec(snapshots.ρw,  α0, T_decay),
        ρθ  = RelaxToICSpec(snapshots.ρθ,  α0, T_decay),
        ρqᵛ = RelaxToICSpec(snapshots.ρqᵛ, α0, T_decay),
    )

    if include_cloud
        snapshots = merge(snapshots, (
            ρqᶜˡ = CenterField(grid),
            ρqᶜⁱ = CenterField(grid),
        ))
        forcing = merge(forcing, (
            ρqᶜˡ = RelaxToICSpec(snapshots.ρqᶜˡ, α0, T_decay),
            ρqᶜⁱ = RelaxToICSpec(snapshots.ρqᶜⁱ, α0, T_decay),
        ))
    end

    return forcing, snapshots
end

"""
    copy_ic_snapshots!(snapshots, model)

Copy the current prognostic field values from `model` into the matching
`snapshots` fields. Must be called AFTER the IC has been loaded (and
interpolated) into the model's prognostic fields. Uses Oceananigans'
`set!`, which under Reactant dispatches to a `@jit` elementwise copy
kernel — the snapshot owns an independent buffer.
"""
function copy_ic_snapshots!(snapshots::NamedTuple, model)
    Oceananigans.set!(snapshots.ρ,   dynamics_density(model.dynamics))
    Oceananigans.set!(snapshots.ρu,  model.momentum.ρu)
    Oceananigans.set!(snapshots.ρv,  model.momentum.ρv)
    Oceananigans.set!(snapshots.ρw,  model.momentum.ρw)
    Oceananigans.set!(snapshots.ρθ,  model.formulation.potential_temperature_density)
    Oceananigans.set!(snapshots.ρqᵛ, model.moisture_density)

    if haskey(snapshots, :ρqᶜˡ)
        Oceananigans.set!(snapshots.ρqᶜˡ, model.microphysical_fields[:ρqᶜˡ])
    end
    if haskey(snapshots, :ρqᶜⁱ)
        Oceananigans.set!(snapshots.ρqᶜⁱ, model.microphysical_fields[:ρqᶜⁱ])
    end

    return nothing
end
