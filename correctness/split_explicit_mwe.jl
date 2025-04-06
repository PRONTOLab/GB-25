using GordonBell25
using KernelAbstractions: @index, @kernel
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

vitd = VerticallyImplicitTimeDiscretization()
vertical_diffusivity = VerticalScalarDiffusivity(vitd, κ=1e-5, ν=1e-4)

kw = (
    #free_surface = SplitExplicitFreeSurface(substeps=2),
    free_surface = ExplicitFreeSurface(),
    coriolis = nothing,
    buoyancy = nothing, # BuoyancyTracer(),
    closure = nothing, # vertical_diffusivity,
    momentum_advection = nothing,
    tracer_advection = nothing,
    Δt = 60,
    halo = (4, 4, 4),
)

Nx = 48
Ny = 24
Nz = 4

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(), Nx, Ny, Nz; kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(), Nx, Ny, Nz; kw...)

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)

@show vmodel
@show rmodel

GordonBell25.sync_states!(rmodel, vmodel)

function problem_kernel!(model)
    free_surface = model.free_surface
    η = free_surface.η
    grid = η.grid
    arch = Oceananigans.architecture(grid)
    g = free_surface.gravitational_acceleration
    timestepper = free_surface.timestepper
    parameters = free_surface.kernel_parameters
    GUⁿ = model.timestepper.Gⁿ.U    
    GVⁿ = model.timestepper.Gⁿ.V
    weights = model.free_surface.substepping.averaging_weights
    Δτᴮ = 1

    U, V = free_surface.barotropic_velocities
    filtered_state = free_surface.filtered_state
    η̅, U̅, V̅ = filtered_state.η, filtered_state.U, filtered_state.V
    U_args = (grid, Δτᴮ, η, U, V, η̅, U̅, V̅, GUⁿ, GVⁿ, g, timestepper)

    barotropic_velocity_kernel!, _ = Oceananigans.Utils.configure_kernel(
        arch, grid, parameters,
        Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces._split_explicit_barotropic_velocity!)

    averaging_weight = weights[1]
    barotropic_velocity_kernel!(averaging_weight, U_args...)

    return nothing
end

Nt = 1
rNt = ConcreteRNumber(Nt)

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)
rupdate! = @compile sync=true raise=true Oceananigans.TimeSteppers.update_state!(rmodel)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)

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

@info "100-step loop!"
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)
GordonBell25.compare_states(rmodel, vmodel)

#=
problem_kernel!(vmodel)
rproblem! = @compile sync=true raise=true problem_kernel!(rmodel)
rproblem!(rmodel)
GordonBell25.compare_states(rmodel, vmodel)


@time rloop!(rmodel, ConcreteRNumber(1))
@time GordonBell25.loop!(vmodel, 1)

@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)

GordonBell25.compare_states(rmodel, vmodel)
=#
