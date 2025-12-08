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

#@jit Oceananigans.initialize!(rmodel)
#Oceananigans.initialize!(vmodel)

using InteractiveUtils
using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.BoundaryConditions: BoundaryCondition, getbc, Flux, Open, _fill_south_and_north_halo!, _fill_south_halo!, _fill_north_halo!, _fill_flux_north_halo!

function my_initialize_free_surface!(sefs, grid, velocities)
    barotropic_velocities = sefs.barotropic_velocities
    u, v, w = velocities
    @apply_regionally Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces.compute_barotropic_mode!(barotropic_velocities.U,
                                               barotropic_velocities.V,
                                               grid, u, v, sefs.η)

    my_tupled_fill_halo_regions!((barotropic_velocities.U, barotropic_velocities.V))

    return nothing
end

function my_tupled_fill_halo_regions!(fields, args...; kwargs...)

    for field in fields
        my_fill_halo_regions!(field.data,
                                field.boundary_conditions,
                                field.indices,
                                Oceananigans.instantiated_location(field),
                                field.grid,
                                args...;
                                reduced_dimensions = (3,),
                                kwargs...)
    end

    return nothing
end

function my_fill_halo_regions!(c, boundary_conditions, indices, loc, grid, args...;
                            fill_boundary_normal_velocities = true, kwargs...)

    arch   = grid.architecture
    size   = :xz
    offset = (0, 0)

    launch!(arch, grid, KernelParameters(size, offset), _my_fill_south_and_north_halo!, c, boundary_conditions.north, grid; kwargs...)

    return nothing
end

@kernel function _my_fill_south_and_north_halo!(c, north_bc, grid)
    i, k = @index(Global, NTuple)
    _my_fill_north_halo!(i, k, grid, c, north_bc)
end

@inline _my_fill_north_halo!(i, k, grid, c, ::BoundaryCondition{<:Flux})        = @inbounds c[i, grid.Ny+1, k] = c[i, grid.Ny, k]
@inline _my_fill_north_halo!(i, k, grid, c, bc::BoundaryCondition{<:Open}) = @inbounds c[i, grid.Ny+1, k] = 0.0


@show "Reactant:"
@jit my_initialize_free_surface!(rmodel.free_surface, rmodel.grid, rmodel.velocities)
@show "Vanilla:"
my_initialize_free_surface!(vmodel.free_surface, vmodel.grid, vmodel.velocities)


@info "After initialization:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

#=
@jit Oceananigans.TimeSteppers.update_state!(rmodel)
Oceananigans.TimeSteppers.update_state!(vmodel)

@info "After update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
=#

#=
GordonBell25.sync_states!(rmodel, vmodel)
rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
@showtime rfirst!(rmodel)
@showtime GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)

@info "Warm up:"
@showtime rstep!(rmodel)
@showtime rstep!(rmodel)
@showtime GordonBell25.time_step!(vmodel)
@showtime GordonBell25.time_step!(vmodel)

Nt = 10
@info "Time step with Reactant:"
for _ in 1:Nt
    @showtime rstep!(rmodel)
end

@info "Time step vanilla:"
for _ in 1:Nt
    @showtime GordonBell25.time_step!(vmodel)
end

@info "After $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)
@jit Oceananigans.TimeSteppers.update_state!(rmodel)

@info "After syncing and updating state again:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@showtime rloop!(rmodel, rNt)
@showtime GordonBell25.loop!(vmodel, Nt)

@info "After a loop of $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
=#