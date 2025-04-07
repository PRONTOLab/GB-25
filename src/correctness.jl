using Reactant
using Printf

function compare_fields(name, ψ1, ψ2; rtol=1e-8, atol=sqrt(eps(eltype(ψ1))))
    ψ1 = Array(interior(ψ1))
    ψ2 = Array(interior(ψ2))
    δ = ψ1 .- ψ2
    @printf("(%4s) ψ₁ ≈ ψ₂: %-5s, max|ψ₁|, max|ψ₂|: %.15e, %.15e, max|δ|: %.15e \n",
            name, isapprox(ψ1, ψ2; rtol, atol),
            maximum(abs, ψ1), maximum(abs, ψ2), maximum(abs, δ))
end

function compare_states(m1, m2; rtol=1e-8, atol=sqrt(eps(eltype(m1.grid))))
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        compare_fields(name, Ψ1[name], Ψ2[name]; rtol, atol)

        if !(name ∈ (:w, :η))
            Gⁿ1 = m1.timestepper.Gⁿ
            Gⁿ2 = m2.timestepper.Gⁿ
            compare_fields("Gⁿ.$name", Gⁿ1[name], Gⁿ2[name]; rtol, atol)

            G⁻1 = m1.timestepper.G⁻
            G⁻2 = m2.timestepper.G⁻
            compare_fields("G⁻.$name", G⁻1[name], G⁻2[name]; rtol, atol)
        end
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity
        names = (:κu, :κc, :κe, :Le, :Jᵇ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            compare_fields(name, Φ1[name], Φ2[name]; rtol, atol)
        end
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity
        names = (:κu, :κc, :κe, :κϵ, :Le, :Lϵ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            compare_fields(name, Φ1[name], Φ2[name]; rtol, atol)
        end
    end

    return nothing
end

function sync_states!(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        ψ1 = Ψ1[name]
        ψ2 = Ψ2[name]
        loc = Oceananigans.Fields.location(ψ1)
        ψ2r = Reactant.to_rarray(interior(Ψ2[name]))
        @jit copy_interior!(ψ1, ψ2r)
        #end
    end
    return nothing
end
  
using KernelAbstractions

function copy_interior!(rf, vf)
    grid = rf.grid
    arch = Oceananigans.Architectures.architecture(grid)
    Oceananigans.Utils.launch!(arch, grid, size(rf), _copy_interior!, interior(rf), vf)
    return nothing
end

@kernel function _copy_interior!(rf, vf)
    i, j, k = @index(Global, NTuple)
    @inbounds rf[i, j, k] = vf[i, j, k]
end
