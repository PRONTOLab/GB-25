using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using Reactant

vitd = VerticallyImplicitTimeDiscretization()
vertical_diffusivity = VerticalScalarDiffusivity(κ=1e-5, ν=1e-4)
vertical_diffusivity = CATKEVerticalDiffusivity(ExplicitTimeDiscretization())

kw = (
    halo = (2, 2, 2),
    free_surface = ExplicitFreeSurface(), #SplitExplicitFreeSurface(substeps=20),
    buoyancy = nothing,
    closure = nothing, 
    coriolis = nothing,
    momentum_advection = nothing, #WENOVectorInvariant(),
    tracer_advection = nothing, #WENO(),
    Δt = 60,
)

Nx = Ny = Nz = 2

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(), Nx, Ny, Nz; kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(), Nx, Ny, Nz; kw...)

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

# Everything is kind of correct till here (errors of about 1e-10)

GordonBell25.compare_states(rmodel, vmodel)

function myloop!(model, Nt)
    @trace track_numbers=false for _ = 1:Nt
        Oceananigans.BoundaryConditions.fill_halo_regions!(model.velocities.v)
    end
end

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true myloop!(rmodel, rNt)
@time rloop!(rmodel, rNt)
@time myloop!(vmodel, Nt)

# Correctness does not work on loops apparently (only for w)
GordonBell25.compare_parent("u", rmodel.velocities.u, vmodel.velocities.u)
GordonBell25.compare_parent("v", rmodel.velocities.v, vmodel.velocities.v)
GordonBell25.compare_parent("w", rmodel.velocities.w, vmodel.velocities.w)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)

# Correctness does not work on loops apparently (only for w)
GordonBell25.compare_parent("u", rmodel.velocities.u, vmodel.velocities.u)
GordonBell25.compare_parent("v", rmodel.velocities.v, vmodel.velocities.v)
GordonBell25.compare_parent("w", rmodel.velocities.w, vmodel.velocities.w)

