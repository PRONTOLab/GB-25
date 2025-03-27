using Reactant
using Printf

function print_comparison(name, ψ1, ψ2)
    ψ1 = Array(parent(ψ1))
    ψ2 = Array(parent(ψ2))
    δ = ψ1 .- ψ2
    @printf("(%4s) ψ₁ ≈ ψ₂: %-5s, max|ψ₁|, max|ψ₂|: %.15e, %.15e, max|δ|: %.15e \n",
            name, ψ1 ≈ ψ2, maximum(abs, ψ1), maximum(abs, ψ2), maximum(abs, δ))
end


function compare_states(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        print_comparison(name, Ψ1[name], Ψ2[name])

        if !(name ∈ (:w, :η))
            Gⁿ1 = m1.timestepper.Gⁿ
            Gⁿ2 = m2.timestepper.Gⁿ
            print_comparison("Gⁿ.$name", Gⁿ1[name], Gⁿ2[name]) 

            G⁻1 = m1.timestepper.G⁻
            G⁻2 = m2.timestepper.G⁻
            print_comparison("G⁻.$name", G⁻1[name], G⁻2[name]) 
        end
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity
        names = (:κu, :κc, :κe, :Le, :Jᵇ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            print_comparison(name, Φ1[name], Φ2[name])
        end
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity
        names = (:κu, :κc, :κe, :κϵ, :Le, :Lϵ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            print_comparison(name, Φ1[name], Φ2[name])
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

