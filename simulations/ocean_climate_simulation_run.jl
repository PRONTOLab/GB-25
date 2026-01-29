using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init, gaussian_islands_tripolar_grid
using Oceananigans.Architectures: ReactantState
using Reactant

using ClimaOcean: ocean_simulation

using Oceananigans

using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

using ClimaOcean.Oceans: default_ocean_closure, TwoColorRadiation, default_or_override, BarotropicPotentialForcing, XDirection, YDirection

default_gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration
default_planet_rotation_rate = Oceananigans.defaults.planet_rotation_rate

using InteractiveUtils

using Oceananigans.BoundaryConditions: DefaultBoundaryCondition

using Oceananigans.Models: initialization_update_state!

using Oceananigans.TimeSteppers: update_state!, Clock

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!, update_boundary_conditions!
using Oceananigans.BuoyancyFormulations: compute_buoyancy_gradients!
using Oceananigans.Fields: compute!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Models: update_model_field_time_series!, surface_kernel_parameters, volume_kernel_parameters
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.Utils: KernelParameters

using Oceananigans.Models.HydrostaticFreeSurfaceModels: mask_immersed_model_fields!, diffusivity_kernel_parameters

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: time_step_catke_equation!

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

function my_data_free_ocean_climate_model_init(
    arch=ReactantState();
    # Horizontal resolution
    resolution::Real = 2, # 1/4 for quarter degree
    # Vertical resolution
    Nz::Int = 20, # eventually we want to increase this to between 100-600
    )

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)

    # See visualize_ocean_climate_simulation.jl for information about how to
    # visualize the results of this run.
    Δt = 30
    free_surface = SplitExplicitFreeSurface(substeps=30)
    @allowscalar ocean = my_ocean_simulation(grid; free_surface)

    return ocean
end

function my_ocean_simulation(grid;
                          closure = default_ocean_closure(),
                          tracers = (:T, :S),
                          free_surface = SplitExplicitFreeSurface(substeps=30),
                          timestepper = :SplitRungeKutta3)

    ocean_model = HydrostaticFreeSurfaceModel(grid;
                                              closure,
                                              tracers,
                                              timestepper,
                                              free_surface)

    return ocean_model
end

@info "Generating model..."
vmodel = my_data_free_ocean_climate_model_init(CPU())
rmodel = my_data_free_ocean_climate_model_init(ReactantState())

function my_update_state!(model, grid, callbacks)

    @show @which time_step_catke_equation!(model, model.timestepper)
    time_step_catke_equation!(model, model.timestepper)

    return nothing
end

#time_step_catke_equation!(model, model.timestepper)
@show @which Clock(vmodel.grid)
@show Clock(vmodel.grid)
@show vmodel.clock
@show vmodel.timestepper
@show vmodel.clock.stage
@show vmodel.timestepper.β


@show @which Clock(rmodel.grid)
@show Clock(rmodel.grid)
@show rmodel.clock
@show rmodel.timestepper
@show rmodel.clock.stage
@show rmodel.timestepper.β

@show vmodel.timestepper.β[vmodel.clock.stage]

@show rmodel.timestepper.β[rmodel.clock.stage]