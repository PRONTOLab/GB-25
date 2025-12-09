using GordonBell25
using Oceananigans
using Reactant

if !GordonBell25.is_distributed_env_present()
    using MPI
    MPI.Init()
end

throw_error = true
include_halos = true
rtol = sqrt(eps(Float64))
atol = 0

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

Rx, Ry = GordonBell25.factors(Ndev)

rarch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition = Partition(Rx, Ry, 1)
)

rank = Reactant.Distributed.local_rank()

H = 8
Tx = 64 * Rx
Ty = 64 * Ry
Nz = 16

Nx = Tx - 2H
Ny = Ty - 2H

model_kw = (
    halo = (H, H, H),
    Δt = 1e-9,
    coriolis = nothing,
    # buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState()),
    momentum_advection = nothing,
    tracer_advection = nothing,
)

varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel
@assert rmodel.architecture isa Distributed

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)

@info "At the beginning:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Oceananigans.Models.NonhydrostaticModels.update_hydrostatic_pressure!(
    vmodel.pressure.pHY′, vmodel.architecture, vmodel.grid, vmodel.buoyancy, vmodel.tracers)
@jit Oceananigans.Models.NonhydrostaticModels.update_hydrostatic_pressure!(
    rmodel.pressure.pHY′, rmodel.architecture, rmodel.grid, rmodel.buoyancy, rmodel.tracers)

@info "After updating hydrostatic pressure:"
GordonBell25.compare_interior("pHY′", rmodel.pressure.pHY′, vmodel.pressure.pHY′)
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

#@jit Oceananigans.initialize!(rmodel)
#Oceananigans.initialize!(vmodel)

using InteractiveUtils
using KernelAbstractions
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ
using Oceananigans.Utils: launch!, _launch!, KernelParameters, configure_kernel, interior_work_layout, work_layout, mapped_kernel
using Oceananigans.BoundaryConditions: BoundaryCondition, getbc, Flux, Open, _fill_south_and_north_halo!, _fill_south_halo!, _fill_north_halo!, _fill_flux_north_halo!
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.Grids: get_active_column_map, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, column_depthᶠᶜᵃ, column_depthᶜᶠᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: _compute_barotropic_mode!

using OffsetArrays

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

rdev = ReactantKernelAbstractionsExt.ReactantBackend()
vdev = KernelAbstractions.CPU()

u   = zeros(Float64, 128, 128, 32)
vu  = OffsetArray(u, -7:120, -7:120, -7:24)
vu .= vmodel.velocities.u.data
ru  = deepcopy(rmodel.velocities.u.data) #Reactant.to_rarray(vu) # This doesn't work

U  = zeros(Float64, 128, 128, 1)
vU = OffsetArray(U, -7:120, -7:120, 1:1)
rU = Reactant.to_rarray(vU)

@show maximum(abs.(parent(vu)))
@show maximum(abs.(convert(Array, parent(ru))))
@show maximum(abs.(convert(Array, parent(ru)) - parent(vu)))

@show maximum(abs.(convert(Array, parent(ru)) - convert(Array, parent(rmodel.velocities.u.data))))

@show typeof(ru)
@show typeof(rmodel.velocities.u.data)

@show size(ru)
@show size(rmodel.velocities.u.data)

#vU .= vmodel.free_surface.barotropic_velocities.U.data
#@allowscalar rU .= rmodel.free_surface.barotropic_velocities.U.data


function my_initialize_free_surface!(U, dev, Ny, u)

    my_compute_barotropic_mode!(U, dev, u)
    my_fill_halo_regions!(U, dev, Ny)

    return nothing
end

function my_compute_barotropic_mode!(U, dev, u)

    workgroup = KernelAbstractions.NDIteration.StaticSize{(16, 16)}()
    worksize  = KernelAbstractions.NDIteration.StaticSize{(112, 112)}()

    loop! = _my_compute_barotropic_mode!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(U, u)

    return nothing
end

@kernel function _my_compute_barotropic_mode!(U̅, u)
    i, j = @index(Global, NTuple)

    @inbounds U̅[i, j, 1] = 1000 * u[i, j, 1]
end

function my_fill_halo_regions!(c, dev, Ny)

    workgroup = KernelAbstractions.NDIteration.StaticSize{(16, 16)}()
    worksize  = KernelAbstractions.NDIteration.StaticSize{(112, 1)}()

    loop! = _my_fill_south_and_north_halo!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(c, Ny)

    return nothing
end

@kernel function _my_fill_south_and_north_halo!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, Ny+1, k] = c[i, Ny, k]
end

@show Oceananigans.Architectures.device(rmodel.grid.architecture)
@show @which Oceananigans.Architectures.device(child_architecture(rmodel.grid.architecture))

@show "Reactant:"
@jit my_initialize_free_surface!(rU, rdev, Ny, ru)
@show "Vanilla:"
my_initialize_free_surface!(vU, vdev, Ny, vu)

@show typeof(rmodel.velocities.u.data)
@show typeof(vmodel.velocities.u.data)

@show size(rmodel.velocities.u.data)
@show size(vmodel.velocities.u.data)

@info "After initialization (should be 0, or at most maybe machine precision, but there's a bug):"
rU = convert(Array, parent(rU))
vU = parent(vU)

@show typeof(rU)
@show typeof(vU)

@show maximum(abs.(rU - vU))
