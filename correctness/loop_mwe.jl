using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

vitd = VerticallyImplicitTimeDiscretization()
vertical_diffusivity = VerticalScalarDiffusivity(vitd, κ=1e-5, ν=1e-4)

kw = (
    free_surface = ExplicitFreeSurface(),
    coriolis = nothing,
    buoyancy = nothing,
    closure = nothing,
    momentum_advection = nothing,
    tracer_advection = nothing,
    Δt = 60,
    halo = (4, 4, 4),
)

Nx = 120
Ny = 60
Nz = 8

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(), Nx, Ny, Nz; kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(), Nx, Ny, Nz; kw...)

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)

@show vmodel
@show rmodel

GordonBell25.sync_states!(rmodel, vmodel)

Nt = 1
rNt = ConcreteRNumber(Nt)

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)
rupdate! = @compile sync=true raise=true Oceananigans.TimeSteppers.update_state!(rmodel)

@time rfirst!(rmodel)
@time GordonBell25.first_time_step!(vmodel)

GordonBell25.sync_states!(rmodel, vmodel)
rupdate! = @compile sync=true raise=true Oceananigans.TimeSteppers.update_state!(rmodel)
rupdate!(rmodel)

GordonBell25.compare_states(rmodel, vmodel)

for n = 1:100
    @time rstep!(rmodel)
    @time GordonBell25.time_step!(vmodel)
end

@info "100 steps"
GordonBell25.compare_states(rmodel, vmodel)

#=
@info "100-step loop!"
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)
GordonBell25.compare_states(rmodel, vmodel)
=#

# Segfault:
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@info "100-step loop!"
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)
GordonBell25.compare_states(rmodel, vmodel)

