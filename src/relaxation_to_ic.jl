#####
##### Relaxation-to-initial-condition forcing with linearly-decaying strength
#####

"""
    RelaxToICForcing{Name, F, T}

A discrete-form Oceananigans/Breeze forcing that damps the prognostic field
named `Name` toward a persistent snapshot `ic`:

```
    F(i,j,k) = -α(t) * (ρϕ[i,j,k] - ρϕ_IC[i,j,k])
```

where the damping rate decays linearly from `α0 = 1/τ₀` at `t=0` to zero at
`t = T_decay`, and is clamped to zero thereafter:

```
    α(t) = max(0, α0 * (1 - t / T_decay))
```

The `Name` parameter is a `Symbol` that identifies which prognostic field we
are damping. It is used to index into the `fields` NamedTuple supplied by the
tendency kernel so that this single forcing type works for any prognostic
variable. `Name` is encoded as a type parameter (not a field) so the
compiler can specialize each tendency and unroll the `fields[Name]` lookup.
"""
struct RelaxToICForcing{Name, F, T}
    ic       :: F
    α0       :: T
    T_decay  :: T
end

RelaxToICForcing(name::Symbol, ic, α0::Real, T_decay::Real) =
    let Tpromoted = promote_type(typeof(α0), typeof(T_decay))
        RelaxToICForcing{name, typeof(ic), Tpromoted}(ic, Tpromoted(α0), Tpromoted(T_decay))
    end

@inline function (f::RelaxToICForcing{Name})(i, j, k, grid, clock, fields) where {Name}
    t = clock.time
    # Linearly-decaying damping rate, clamped to zero after T_decay
    α = max(zero(t), f.α0 * (1 - t / f.T_decay))
    @inbounds ic_val  = f.ic[i, j, k]
    @inbounds cur_val = fields[Name][i, j, k]
    return -α * (cur_val - ic_val)
end

"""
    build_ic_relaxation_forcing(grid; α0, T_decay, include_cloud=false)

Allocate IC-snapshot fields and build a NamedTuple of `RelaxToICForcing`s
that damp each of the six dynamics/vapor prognostic fields (and optionally
the two cloud-condensate fields) toward those snapshots.

`α0` is the initial damping rate (s⁻¹); the rate decays linearly to zero at
`t = T_decay` and is clamped to zero thereafter.

The snapshot fields are returned alongside the forcing NamedTuple so the
caller can fill them with post-interpolation state after the IC is loaded.
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
        ρ   = RelaxToICForcing(:ρ,   snapshots.ρ,   α0, T_decay),
        ρu  = RelaxToICForcing(:ρu,  snapshots.ρu,  α0, T_decay),
        ρv  = RelaxToICForcing(:ρv,  snapshots.ρv,  α0, T_decay),
        ρw  = RelaxToICForcing(:ρw,  snapshots.ρw,  α0, T_decay),
        ρθ  = RelaxToICForcing(:ρθ,  snapshots.ρθ,  α0, T_decay),
        ρqᵛ = RelaxToICForcing(:ρqᵛ, snapshots.ρqᵛ, α0, T_decay),
    )

    if include_cloud
        snapshots = merge(snapshots, (
            ρqᶜˡ = CenterField(grid),
            ρqᶜⁱ = CenterField(grid),
        ))
        forcing = merge(forcing, (
            ρqᶜˡ = RelaxToICForcing(:ρqᶜˡ, snapshots.ρqᶜˡ, α0, T_decay),
            ρqᶜⁱ = RelaxToICForcing(:ρqᶜⁱ, snapshots.ρqᶜⁱ, α0, T_decay),
        ))
    end

    return forcing, snapshots
end

"""
    copy_ic_snapshots!(snapshots, model)

Copy the current prognostic field values from `model` into the matching
`snapshots` fields. Must be called AFTER the IC has been loaded (and
interpolated) into the model's prognostic fields. Uses Oceananigans' `set!`
which under Reactant dispatches to a `@jit` elementwise copy kernel,
producing an independent buffer (so the snapshot does not alias the
prognostic state).
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
