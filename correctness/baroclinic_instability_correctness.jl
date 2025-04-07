using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

H = 8 # halo size
Tx, Ty = 128, 64
Nx, Ny = (Tx, Ty) .- 2H
Nz = 32

vitd = VerticallyImplicitTimeDiscretization()
vertical_diffusivity = VerticalScalarDiffusivity(vitd, κ=1e-5, ν=1e-4)

kw = (
    resolution = 2,
    #free_surface = SplitExplicitFreeSurface(substeps=2),
    free_surface = ExplicitFreeSurface(),
    coriolis = nothing,
    # buoyancy = nothing, # BuoyancyTracer(),
    buoyancy = BuoyancyTracer(),
    closure = nothing, # vertical_diffusivity,
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

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)

@time rfirst!(rmodel)
@time GordonBell25.first_time_step!(vmodel)
GordonBell25.compare_states(rmodel, vmodel)

#=
Nt = 10
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)
GordonBell25.compare_states(rmodel, vmodel)
=#
