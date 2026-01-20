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

using Oceananigans.Fields: tracernames
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Models: extract_boundary_conditions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: constructor_field_names, hydrostatic_velocity_fields
using Oceananigans.TurbulenceClosures: add_closure_specific_boundary_conditions
using Oceananigans.TimeSteppers: Clock

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
    free_surface = SplitExplicitFreeSurface(substeps=30)
    tracers = (:T, :S)

    #=
    # We need boundary_conditions:
    embedded_boundary_conditions = merge(extract_boundary_conditions(nothing),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(nothing),
                                         extract_boundary_conditions(nothing))

    # Next, we form a list of default boundary conditions:
    field_names = constructor_field_names(nothing, tracers, free_surface, NamedTuple(), nothing, grid)
    default_boundary_conditions = NamedTuple{field_names}(FieldBoundaryConditions() for name in field_names)

    # Then we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    boundary_conditions = merge(default_boundary_conditions, embedded_boundary_conditions, nothing)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, field_names)

    # Finally, we ensure that closure-specific boundary conditions, such as
    # those required by CATKEVerticalDiffusivity, are enforced:
    boundary_conditions = add_closure_specific_boundary_conditions(nothing,
                                                                   boundary_conditions,
                                                                   grid,
                                                                   tracernames(tracers),
                                                                   nothing)

    # We need velocities:
    velocities = hydrostatic_velocity_fields(nothing, grid, Clock(grid), boundary_conditions)

    # Problem line:
    free_surface = materialize_free_surface(free_surface, velocities, grid)

    =#
    @allowscalar ocean = HydrostaticFreeSurfaceModel(; grid,
                                                        tracers,
                                                        free_surface)

    return ocean
end

@info "Generating model..."
model = my_data_free_ocean_climate_model_init(ReactantState())