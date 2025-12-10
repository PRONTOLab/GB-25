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

@jit Oceananigans.initialize!(rmodel, rmodel.grid)
Oceananigans.initialize!(vmodel, vmodel.grid)

using InteractiveUtils

using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_tendency_contributions!, compute_hydrostatic_momentum_tendencies!





#@show @which Oceananigans.TimeSteppers.compute_tendencies!(rmodel, [])
#@show @which Oceananigans.TimeSteppers.compute_tendencies!(vmodel, [])

#@show @which compute_hydrostatic_free_surface_tendency_contributions!(rmodel, :xyz)
#@show @which compute_hydrostatic_free_surface_tendency_contributions!(vmodel, :xyz)

@show @which compute_hydrostatic_momentum_tendencies!(rmodel, rmodel.velocities, :xyz)
@show @which compute_hydrostatic_momentum_tendencies!(vmodel, vmodel.velocities, :xyz)


@jit compute_hydrostatic_momentum_tendencies!(rmodel, rmodel.velocities, :xyz)
compute_hydrostatic_momentum_tendencies!(vmodel, vmodel.velocities, :xyz)

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

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