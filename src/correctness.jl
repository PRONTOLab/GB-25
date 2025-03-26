using Reactant

function compare_states(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        ψ1p = Array(parent(Ψ1[name]))
        ψ2p = Array(parent(Ψ2[name]))
        @show name, ψ1p ≈ ψ2p
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

