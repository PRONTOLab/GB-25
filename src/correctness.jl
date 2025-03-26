using Reactant

function compare_states(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        ψ1p = Array(parent(Ψ1[name]))
        ψ2p = Array(parent(Ψ2[name]))
        δ = ψ1p .- ψ2p
        @show name, ψ1p ≈ ψ2p, maximum(abs, ψ1p), maximum(ψ2p), maximum(abs, δ)
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity
        names = (:κu, :κc, :κe, :Le, :Jᵇ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            ϕ1p = Array(parent(Φ1[name]))
            ϕ2p = Array(parent(Φ2[name]))
            δ = ϕ1p .- ϕ2p
            @show name, ϕ1p ≈ ϕ2p, maximum(abs, ϕ1p), maximum(ϕ2p), maximum(abs, δ)
        end
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity
        names = (:κu, :κc, :κe, :κϵ, :Le, :Lϵ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            ϕ1p = Array(parent(Φ1[name]))
            ϕ2p = Array(parent(Φ2[name]))
            δ = ϕ1p .- ϕ2p
            @show name, ϕ1p ≈ ϕ2p, maximum(abs, ϕ1p), maximum(ϕ2p), maximum(abs, δ)
        end
    end

    return nothing
end

function sync_states!(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        ψ1p = parent(Ψ1[name])
        ψ2p = parent(Ψ2[name])
        copyto!(ψ1p, Reactant.to_rarray(ψ2p))
    end
    return nothing
end

