using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

H = 8 # halo size
Tx, Ty = 128, 64
Nx, Ny = (Tx, Ty) .- 2H
Nz = 32

kw = (
    resolution = 8,
    free_surface = SplitExplicitFreeSurface(substeps=32),
    #free_surface = ExplicitFreeSurface(),
    coriolis = nothing,
    buoyancy = BuoyancyTracer(),
    closure = nothing,
    momentum_advection = nothing,
    tracer_advection = nothing,
    Δt = 60,
    Nz = 10,
)

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(); kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(); kw...)

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)

@show vmodel
@show rmodel

GordonBell25.sync_states!(rmodel, vmodel)
GordonBell25.compare_states(rmodel, vmodel)

function problem_kernel(model)
    Oceananigans.Models.HydrostaticFreeSurfaceModels.update_hydrostatic_pressure!(
        model.pressure.pHY′,
        model.grid.architecture,
        model.grid,
        model.buoyancy,
        model.tracers
    )
    return nothing
end

rproblem = @compile sync=true raise=true problem_kernel(rmodel)
rproblem(rmodel)
problem_kernel(vmodel)

rp = rmodel.pressure.pHY′
vp = vmodel.pressure.pHY′
GordonBell25.compare_fields("pHY", rp, vp)

#=
Nt = 10
rNt = ConcreteRNumber(Nt)

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)

@time rfirst!(rmodel)
@time GordonBell25.first_time_step!(vmodel)
GordonBell25.compare_states(rmodel, vmodel)

# rloop!(rmodel, rNt)
# @time GordonBell25.loop!(rmodel, Nt)
# @time GordonBell25.compare_states(rmodel, vmodel)
=#
