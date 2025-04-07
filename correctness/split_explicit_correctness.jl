using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

using Oceananigans.Grids: architecture
using Oceananigans.Operators: ∂xTᶠᶜᶠ, ∂yTᶜᶠᶠ, ∂xᶠᶜᶠ, ∂yᶜᶠᶠ
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: η★

using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

kw = (
    resolution = 2,
    free_surface = SplitExplicitFreeSurface(substeps=10),
    coriolis = nothing,
    buoyancy = nothing, # BuoyancyTracer(),
    closure = nothing, # vertical_diffusivity,
    momentum_advection = nothing,
    tracer_advection = nothing,
    Δt = 60,
    Nz = 10,
)

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(); kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(); kw...)

ηi = rand(size(interior(vmodel.free_surface.η))...)
Ui = rand(size(interior(vmodel.free_surface.barotropic_velocities.U))...)
Vi = rand(size(interior(vmodel.free_surface.barotropic_velocities.V))...)

interior(vmodel.free_surface.η) .= ηi
interior(vmodel.free_surface.barotropic_velocities.U) .= Ui
interior(vmodel.free_surface.barotropic_velocities.V) .= Vi

Oceananigans.fill_halo_regions!(vmodel.free_surface.η)
Oceananigans.fill_halo_regions!(vmodel.free_surface.barotropic_velocities.U)
Oceananigans.fill_halo_regions!(vmodel.free_surface.barotropic_velocities.V)

GordonBell25.sync_states!(rmodel, vmodel)

@jit Oceananigans.fill_halo_regions!(rmodel.free_surface.η)
@jit Oceananigans.fill_halo_regions!(rmodel.free_surface.barotropic_velocities.U)
@jit Oceananigans.fill_halo_regions!(rmodel.free_surface.barotropic_velocities.V)
# The two models should be identical from here on.
GordonBell25.compare_states(rmodel, vmodel)

rsefs = rmodel.free_surface
vsefs = vmodel.free_surface
vts   = vmodel.timestepper

@show rsefs.substepping == vsefs.substepping

rgrid = rsefs.η.grid
rη  = rsefs.η
rU  = rsefs.barotropic_velocities.U
rV  = rsefs.barotropic_velocities.V
rU̅  = rsefs.filtered_state.U
rV̅  = rsefs.filtered_state.V
rη̅  = rsefs.filtered_state.η
rargs = (rgrid, rη, rU, rV, rη̅, rU̅, rV̅)

vgrid = vsefs.η.grid
vη  = vsefs.η
vU  = vsefs.barotropic_velocities.U
vV  = vsefs.barotropic_velocities.V
vU̅  = vsefs.filtered_state.U
vV̅  = vsefs.filtered_state.V
vη̅  = vsefs.filtered_state.η
vargs = (vgrid, vη, vU, vV, vη̅, vU̅, vV̅)

GordonBell25.compare_parent_fields("η", rη, vη)
GordonBell25.compare_parent_fields("U", rU, vU)
GordonBell25.compare_interior_fields("η", rη, vη)
GordonBell25.compare_interior_fields("U", rU, vU)

@kernel function _split_explicit_barotropic_velocity!(averaging_weight, grid,  
                                                      η, U, V, 
                                                      η̅, U̅, V̅)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1
    
    @inbounds begin
        # ∂τ(U) = - ∇η + G
        U[i, j, 1] += η[i, j, k_top] - η[i-1, j,   k_top]
        V[i, j, 1] += η[i, j, k_top] - η[i,   j-1, k_top]

        # time-averaging
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        U̅[i, j, 1]     += averaging_weight * U[i, j, 1]
        V̅[i, j, 1]     += averaging_weight * V[i, j, 1]
    end
end

function bad_kernel_launch!(grid, averaging_weight, args)
    arch = architecture(grid)
    dev = Oceananigans.Architectures.device(arch)
    workgroup = (16, 16)
    worksize = (size(grid, 1), size(grid, 2))
    _split_explicit_barotropic_velocity!(dev, workgroup, worksize)(averaging_weight, args...)
end

averaging_weight = 0.5

rkernel! = @compile sync=true raise=true bad_kernel_launch!(rgrid, averaging_weight, rargs)
@time rkernel!(rgrid, averaging_weight, rargs)
@time bad_kernel_launch!(vgrid, averaging_weight, vargs)

GordonBell25.compare_interior_fields("U", rU, vU)
GordonBell25.compare_interior_fields("V", rV, vV)
GordonBell25.compare_interior_fields("U̅", rU̅, vU̅)
GordonBell25.compare_interior_fields("V̅", rV̅, vV̅)