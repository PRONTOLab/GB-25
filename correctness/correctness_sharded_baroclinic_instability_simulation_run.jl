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

    my_fill_reduced_field_halos!(fields, args...; kwargs)

    return nothing
end

# Helper function to create the tuple of ordinary fields:
function my_fill_reduced_field_halos!(fields, args...; kwargs)

    not_reduced_fields = Field[]
    for f in fields
        Oceananigans.BoundaryConditions.fill_halo_regions!(f, args...; kwargs...)
    end

    return nothing
end

@jit my_initialize_free_surface!(rmodel.free_surface, rmodel.grid, rmodel.velocities)
my_initialize_free_surface!(vmodel.free_surface, vmodel.grid, vmodel.velocities)

using InteractiveUtils

#@show @which Oceananigans.Fields.fill_reduced_field_halos!((rmodel.free_surface.barotropic_velocities.U, rmodel.free_surface.barotropic_velocities.V))
#@show @which Oceananigans.Fields.fill_reduced_field_halos!((vmodel.free_surface.barotropic_velocities.U, vmodel.free_surface.barotropic_velocities.V))


@show @which Oceananigans.BoundaryConditions.fill_halo_regions!(rmodel.free_surface.barotropic_velocities.U)
@show @which Oceananigans.BoundaryConditions.fill_halo_regions!(vmodel.free_surface.barotropic_velocities.U)

@show @which Oceananigans.BoundaryConditions.fill_halo_regions!(rmodel.free_surface.barotropic_velocities.V)
@show @which Oceananigans.BoundaryConditions.fill_halo_regions!(vmodel.free_surface.barotropic_velocities.V)

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