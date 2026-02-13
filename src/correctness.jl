using Reactant
using Printf

function compare_parent(name, ψ1, ψ2; rtol=1e-8, atol=sqrt(eps(eltype(ψ1))))
    ψ1 = Array(parent(ψ1))
    ψ2 = Array(parent(ψ2))
    Nx, Ny, Nz = size(ψ1)
    ψ2 = view(ψ2, 1:Nx, 1:Ny, 1:Nz) # assuming that ψ1 is smaller than ψ2
    δ = ψ1 .- ψ2
    idxs = findmax(abs, δ)[2]
    approx_equal = isapprox(ψ1, ψ2; rtol, atol)
    @printf("(%4s) ψ₁ ≈ ψ₂: %-5s, max|ψ₁|, max|ψ₂|: %.15e, %.15e, max|δ|: %.15e at %d %d %d \n",
            name, approx_equal, maximum(abs, ψ1), maximum(abs, ψ2), maximum(abs, δ), idxs.I...)
    return approx_equal
end

function compare_interior(name, ψ1, ψ2; rtol=1e-8, atol=sqrt(eps(eltype(ψ1))))
    ψ1 = Array(interior(ψ1))
    ψ2 = Array(interior(ψ2))
    δ = ψ1 .- ψ2
    idxs = findmax(abs, δ)[2]
    approx_equal = isapprox(ψ1, ψ2; rtol, atol)
    @printf("(%4s) ψ₁ ≈ ψ₂: %-5s, max|ψ₁|, max|ψ₂|: %.15e, %.15e, max|δ|: %.15e at %d %d %d \n",
            name, approx_equal, maximum(abs, ψ1), maximum(abs, ψ2), maximum(abs, δ), idxs.I...)
    return approx_equal
end

function compare_states(m1, m2; rtol=1e-8, atol=sqrt(eps(eltype(m1.grid))),
                        include_halos=false, throw_error=false)

    approx_equal = true
    compare_fields = include_halos ? compare_parent : compare_interior

    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)

    for name in keys(Ψ1)
        approx_equal *= compare_fields(name, Ψ1[name], Ψ2[name]; rtol, atol)

        if !(name ∈ (:w, :η))
            Gⁿ1 = m1.timestepper.Gⁿ
            Gⁿ2 = m2.timestepper.Gⁿ
            approx_equal *= compare_fields("Gⁿ.$name", Gⁿ1[name], Gⁿ2[name]; rtol, atol)

            G⁻1 = m1.timestepper.G⁻
            G⁻2 = m2.timestepper.G⁻
            approx_equal *= compare_fields("G⁻.$name", G⁻1[name], G⁻2[name]; rtol, atol)
        end
    end

    if m1.free_surface isa Oceananigans.SplitExplicitFreeSurface
        names = (:U̅, :V̅, :Ũ, :Ṽ, :η̅)
        Φ1 = NamedTuple(name => getproperty(m1.free_surface.filtered_state, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.free_surface.filtered_state, name) for name in names)
        for name in keys(Φ1)
            approx_equal *= compare_fields(name, Φ1[name], Φ2[name]; rtol, atol)
        end
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity
        names = (:κu, :κc, :κe, :Le, :Jᵇ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            approx_equal *= compare_fields(name, Φ1[name], Φ2[name]; rtol, atol)
        end
    end

    if m1.closure isa Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity
        names = (:κu, :κc, :κe, :κϵ, :Le, :Lϵ)
        Φ1 = NamedTuple(name => getproperty(m1.diffusivity_fields, name) for name in names)
        Φ2 = NamedTuple(name => getproperty(m2.diffusivity_fields, name) for name in names)
        for name in keys(Φ1)
            approx_equal *= compare_fields(name, Φ1[name], Φ2[name]; rtol, atol)
        end
    end

    if approx_equal
        @info "The two models are consistent within rtol=$(rtol) and atol=$(atol) !"
    else
        err_msg = "There is a discrepancy between the models!  See the details above"
        if throw_error
            error(err_msg)
        else
            @error err_msg
        end
    end

    return nothing
end

function sync_states!(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        ψ1 = parent(Ψ1[name])
        x, y, z = size(ψ1)
        ψ2 = parent(Ψ2[name])
        ψ2 = view(ψ2, 1:x, 1:y, 1:z)
        copyto!(ψ1, Array(ψ2))
    end
    return nothing
end

function sync_parent_states!(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        ψ1p = parent(Ψ1[name])
        ψ2p = parent(Ψ2[name])
        copyto!(ψ1p, Reactant.to_rarray(ψ2p))
    end

    return nothing
end

using KernelAbstractions

function copy_interior!(rf, vf, grid)
    arch = Oceananigans.Architectures.architecture(grid)
    Oceananigans.Utils.launch!(arch, grid, size(rf), _copy_interior!, rf, vf)
    return nothing
end

@kernel function _copy_interior!(rf, vf)
    i, j, k = @index(Global, NTuple)
    @inbounds rf[i, j, k] = vf[i, j, k]
end
