using GordonBell25
using Oceananigans
using Reactant

kw = (
    halo = (6, 6, 6),
    Δt = 60,
)

Nx = Ny = Nz = 8
rarch = Oceananigans.Architectures.ReactantState()
varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; kw...)
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
GordonBell25.compare_states(rmodel, vmodel, throw_error=true)

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
@time rfirst!(rmodel)
@time GordonBell25.first_time_step!(vmodel)
GordonBell25.compare_states(rmodel, vmodel, throw_error=true)

rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)

@info "Warm up:"
@time rstep!(rmodel)
@time rstep!(rmodel)
@time GordonBell25.time_step!(vmodel)
@time GordonBell25.time_step!(vmodel)

@info "Time step with Reactant:"
for _ in 1:10
    @time rstep!(rmodel)
end

@info "Time step vanilla:"
for _ in 1:10
    @time GordonBell25.time_step!(vmodel)
end

GordonBell25.compare_states(rmodel, vmodel, include_halos=true, throw_error=true)

GordonBell25.sync_states!(rmodel, vmodel)
@jit Oceananigans.TimeSteppers.update_state!(rmodel)
GordonBell25.compare_states(rmodel, vmodel, include_halos=true, throw_error=true)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)

GordonBell25.compare_states(rmodel, vmodel, include_halos=true, throw_error=true)

