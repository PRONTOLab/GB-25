using GordonBell25
using Oceananigans
using Reactant

throw_error = true
include_halos = true
rtol = 0
atol = 1e-7

model_kw = (
    halo = (6, 6, 6),
    Δt = 1e-9,
)

Nx = Ny = Nz = 16
rarch = Oceananigans.Architectures.ReactantState()
varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)

@jit Oceananigans.initialize!(rmodel)
Oceananigans.initialize!(vmodel)

@jit Oceananigans.TimeSteppers.update_state!(rmodel)
Oceananigans.TimeSteppers.update_state!(vmodel)

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)
rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
@time rfirst!(rmodel)
@time GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)

@info "Warm up:"
@time rstep!(rmodel)
@time rstep!(rmodel)
@time GordonBell25.time_step!(vmodel)
@time GordonBell25.time_step!(vmodel)

Nt = 10
@info "Time step with Reactant:"
for _ in 1:Nt
    @time rstep!(rmodel)
end

@info "Time step vanilla:"
for _ in 1:Nt
    @time GordonBell25.time_step!(vmodel)
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
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)

@info "After a loop of $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
