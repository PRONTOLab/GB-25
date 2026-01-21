using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init, gaussian_islands_tripolar_grid
using Oceananigans.Architectures: ReactantState
using Reactant

using InteractiveUtils

using ClimaOcean: ocean_simulation

using Oceananigans

using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

using ClimaOcean.Oceans: default_ocean_closure, TwoColorRadiation, default_or_override, BarotropicPotentialForcing, XDirection, YDirection

default_gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration
default_planet_rotation_rate = Oceananigans.defaults.planet_rotation_rate

using Oceananigans.Fields: tracernames
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, FieldBoundaryConditions, regularize_field_boundary_conditions
using Oceananigans.Models: extract_boundary_conditions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: constructor_field_names, hydrostatic_velocity_fields, materialize_free_surface
using Oceananigans.TurbulenceClosures: add_closure_specific_boundary_conditions
using Oceananigans.TimeSteppers: Clock

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos

using Oceananigans.Grids: topology, halo_size, LeftConnected, RightConnected, FullyConnected, with_halo

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

const ConnectedTopology = Union{LeftConnected, RightConnected, FullyConnected}

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

    # Next, we form a list of default boundary conditions:
    field_names = constructor_field_names(nothing, tracers, free_surface, NamedTuple(), nothing, grid)
    boundary_conditions = NamedTuple{field_names}(FieldBoundaryConditions() for name in field_names)

    # Then we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    @show @which regularize_field_boundary_conditions(boundary_conditions, grid, field_names)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, field_names)

    @show boundary_conditions

    # We need velocities:
    @show @which hydrostatic_velocity_fields(nothing, grid, Clock(grid), boundary_conditions)
    velocities = hydrostatic_velocity_fields(nothing, grid, Clock(grid), boundary_conditions)

    # Problem line:
    @show @which materialize_free_surface(free_surface, velocities, grid)
    @allowscalar free_surface = my_materialize_free_surface(free_surface, velocities, grid)

    return ocean
end

function my_materialize_free_surface(free_surface, velocities, grid)

    TX, TY, _   = topology(grid)
    substepping = free_surface.substepping

    @show @which maybe_extend_halos(TX, TY, grid, substepping)
    maybe_extended_grid = my_maybe_extend_halos(TX, TY, grid, substepping)

end

function my_maybe_extend_halos(TX, TY, grid, substepping)

    old_halos = halo_size(grid)
    Nsubsteps = length(substepping.averaging_weights)

    Hx = TX() isa ConnectedTopology ? max(Nsubsteps+2, old_halos[1]) : old_halos[1]
    Hy = TY() isa ConnectedTopology ? max(Nsubsteps+2, old_halos[2]) : old_halos[2]

    new_halos = (Hx, Hy, old_halos[3])

    if new_halos == old_halos
        return grid
    else
        return with_halo(new_halos, grid)
    end
end

@info "Generating model..."
model = my_data_free_ocean_climate_model_init(ReactantState())