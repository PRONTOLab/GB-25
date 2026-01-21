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

using Oceananigans.Fields: tracernames, set_to_field!
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Models: extract_boundary_conditions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: constructor_field_names, hydrostatic_velocity_fields, materialize_free_surface
using Oceananigans.TurbulenceClosures: add_closure_specific_boundary_conditions
using Oceananigans.TimeSteppers: Clock

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos

using Oceananigans.Grids: topology, halo_size, LeftConnected, RightConnected, FullyConnected, with_halo

using Oceananigans.ImmersedBoundaries: has_active_cells_map, has_active_z_columns, materialize_immersed_boundary, compute_numerical_bottom_height!, GridFittedBottom

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

const ConnectedTopology = Union{LeftConnected, RightConnected, FullyConnected}

function my_data_free_ocean_climate_model_init(arch;
    # Horizontal resolution
    resolution::Real = 2, # 1/4 for quarter degree
    # Vertical resolution
    Nz::Int = 20, # eventually we want to increase this to between 100-600
    )

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)

    # Problem line:
    #@allowscalar free_surface = my_materialize_free_surface(grid)

    return grid
end

function my_materialize_free_surface(grid)

    old_halos = halo_size(grid)

    Hx = 8
    Hy = 23

    new_halos = (Hx, Hy, old_halos[3])

    my_with_halo(new_halos, grid)
end

function my_with_halo(halo, ibg)
    underlying_grid = with_halo(halo, ibg.underlying_grid)

    return underlying_grid
end

function my_materialize_immersed_boundary(grid, ib)
    bottom_field = Field{Center, Center, Nothing}(grid)

    @show @which parent(bottom_field.data)
    @show typeof(parent(bottom_field.data))
    @show size(parent(bottom_field.data))
    @show typeof(bottom_field.data)
    @show size(bottom_field.data)

    @show @which parent(ib.bottom_height.data)
    @show typeof(parent(ib.bottom_height.data))
    @show size(parent(ib.bottom_height.data))
    @show typeof(ib.bottom_height.data)
    @show size(ib.bottom_height.data)

    parent(bottom_field.data) .= parent(ib.bottom_height.data)
    return new_ib
end

@info "Generating model..."
grid = my_data_free_ocean_climate_model_init(ReactantState())

@allowscalar underlying_grid = my_materialize_free_surface(grid)

materialized_ib = my_materialize_immersed_boundary(underlying_grid, grid.immersed_boundary)