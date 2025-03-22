using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
    calculate_substeps,
    calculate_adaptive_settings

function simple_model(arch)

    Nx = Ny = Nz = 16
    grid = LatitudeLongitudeGrid(arch,
                                 topology = (Periodic, Bounded, Bounded),
                                 size = (Nx, Ny, Nz),
                                 longitude = (-10, 10),
                                 latitude = (40, 60),
                                 z = (-1000, 0),
                                 halo = (6, 6, 6))

    momentum_advection = WENOVectorInvariant() # doesn't work
    #momentum_advection = VectorInvariant() # works
    
    #free_surface = SplitExplicitFreeSurface(substeps=10)
    free_surface = ExplicitFreeSurface()

    model = HydrostaticFreeSurfaceModel(; grid, free_surface, momentum_advection)
    model.clock.last_Δt = 60

    return model
end

c_model = simple_model(CPU())
r_model = simple_model(Oceananigans.Architectures.ReactantState())

using Random
Random.seed!(123)
puc = parent(c_model.velocities.u)
pur = parent(r_model.velocities.u)
pvc = parent(c_model.velocities.v)
pvr = parent(r_model.velocities.v)
pwc = parent(c_model.velocities.w)
pwr = parent(r_model.velocities.w)
u₀ = rand(size(puc)...)
v₀ = rand(size(pvc)...)
w₀ = rand(size(pwc)...)

# NB: The correctness issue requires setting v, w and seems to be triggered
# by either WENOVectorInvariant or WENO. This suggests that the code
# generated within `advective_momentum_flux_Wv` is problematic.
#copyto!(puc, u₀)
copyto!(pvc, v₀)
copyto!(pwc, w₀)
#copyto!(pur, Reactant.to_rarray(u₀))
copyto!(pvr, Reactant.to_rarray(v₀))
copyto!(pwr, Reactant.to_rarray(w₀))

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    interior_tendency_kernel_parameters,
    immersed_boundary_condition,
    get_active_cells_map,
    compute_hydrostatic_free_surface_Gu!,
    compute_hydrostatic_free_surface_Gv!

using Oceananigans.Utils: launch!
using Oceananigans.Operators: ℑzᵃᵃᶜ, Azᶜᶠᶜ, δzᵃᵃᶜ, Az_qᶜᶜᶠ, ℑyᵃᶠᵃ
using Oceananigans.Advection: Vᶜᶠᶜ, bias
using KernelAbstractions: @index, @kernel

@inline function fake_momentum_flux_Wv_symmetric(i, j, k, grid, scheme, W, v)
    #w̃  = Oceananigans.Advection._symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, Az_qᶜᶜᶠ, W)
    w̃  = Oceananigans.Advection._symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    #return w̃ * vᴿ
    return w̃
end

@inline function fake_momentum_flux_Wv_biased(i, j, k, grid, scheme, W, v)
    #w̃ = -1 #@inbounds W[i, j, k]
    #w̃  = Oceananigans.Advection._symmetric_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, W)
    w̃  = ℑyᵃᶠᵃ(i, j, k, grid, W)
    vᴿ = Oceananigans.Advection._biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias(w̃), v)
    return vᴿ
end

@inline function fake_vertical_advection_V(i, j, k, grid, scheme::VectorInvariant, U)
    𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, fake_momentum_flux_Wv_biased, scheme.vertical_scheme, U.w, U.v)
    #𝒜ᶻ = δzᵃᵃᶜ(i, j, k, grid, fake_momentum_flux_Wv_symmetric, scheme.vertical_scheme, U.w, U.v)
    return 𝒜ᶻ
end

@kernel function _problem_kernel!(Gv, grid, advection, velocities)
    i, j, k = @index(Global, NTuple)
    #@inbounds Gv[i, j, k] = - Oceananigans.Advection.U_dot_∇v(i, j, k, grid, advection, velocities)
    #@inbounds Gv[i, j, k] = - Oceananigans.Advection.vertical_advection_V(i, j, k, grid, advection, velocities)
    @inbounds Gv[i, j, k] = - fake_vertical_advection_V(i, j, k, grid, advection, velocities)
end

function launch_problem_kernel!(model)
    grid = model.grid
    arch = Oceananigans.Architectures.architecture(grid)
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)
    launch!(arch, grid, kernel_parameters,
            _problem_kernel!,
            model.timestepper.Gⁿ.v, grid, model.advection.momentum, model.velocities)

    return nothing
end

passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for" #,canonicalize,affine-cfg" #,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion,affine-loop-unroll{unroll-full}),canonicalize,raise-affine-to-stablehlo,arith-raise{stablehlo=true}"

r_problem! = @compile sync=true raise=true launch_problem_kernel!(r_model)
#r_problem! = @compile sync=true raise=false launch_problem_kernel!(r_model)
@time r_problem!(r_model)
@time launch_problem_kernel!(c_model)

function compare(c, r, name="")
    pc = Array(parent(c))
    pr = Array(parent(r))
    @show name, maximum(pc .- pr)
end

Gvc = c_model.timestepper.Gⁿ.v
Gvr = r_model.timestepper.Gⁿ.v
compare(Gvc, Gvr, "Gv")

