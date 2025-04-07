using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using Reactant

H = 8 # halo size
Tx, Ty = 128, 64
Nx, Ny = (Tx, Ty) .- 2H
Nz = 32

vitd = VerticallyImplicitTimeDiscretization()
vertical_diffusivity = VerticalScalarDiffusivity(κ=1e-5, ν=1e-4)
vertical_diffusivity = CATKEVerticalDiffusivity(ExplicitTimeDiscretization())

kw = (
    resolution = 2,
    free_surface = SplitExplicitFreeSurface(substeps=10),
    coriolis = nothing,
    buoyancy = BuoyancyTracer(),
    closure = vertical_diffusivity,
    momentum_advection = WENOVectorInvariant(),
    tracer_advection = WENO(),
    Δt = 60,
    Nz = 10,
)

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(); kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(); kw...)

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)

@jit Oceananigans.initialize!(rmodel)
Oceananigans.initialize!(vmodel)

@jit Oceananigans.TimeSteppers.update_state!(rmodel)
Oceananigans.TimeSteppers.update_state!(vmodel)


GordonBell25.sync_states!(rmodel, vmodel)
GordonBell25.compare_states(rmodel, vmodel)

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)

@time rfirst!(rmodel)
@time GordonBell25.first_time_step!(vmodel)
GordonBell25.compare_states(rmodel, vmodel)

rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)

for _ in 1:10
    @time rstep!(rmodel)
    @time GordonBell25.time_step!(vmodel)
end

# Everything is kind of correct till here (errors of about 1e-10)

GordonBell25.compare_states(rmodel, vmodel)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)

# Correctness does not work on loops apparently (only for w)

GordonBell25.compare_states(rmodel, vmodel)

