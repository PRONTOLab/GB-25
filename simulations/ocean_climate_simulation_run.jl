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

using Oceananigans.Fields: tracernames, set_to_field!, location
using Oceananigans.BoundaryConditions: DefaultBoundaryCondition, FieldBoundaryConditions, regularize_field_boundary_conditions, fill_halo_regions!
using Oceananigans.Models: extract_boundary_conditions
using Oceananigans.Models.HydrostaticFreeSurfaceModels: constructor_field_names, hydrostatic_velocity_fields, materialize_free_surface
using Oceananigans.TurbulenceClosures: add_closure_specific_boundary_conditions
using Oceananigans.TimeSteppers: Clock

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos

using Oceananigans.Grids: topology, halo_size, LeftConnected, RightConnected, FullyConnected, with_halo

using Oceananigans.ImmersedBoundaries: has_active_cells_map, has_active_z_columns, materialize_immersed_boundary, compute_numerical_bottom_height!, GridFittedBottom

using Oceananigans.DistributedComputations: child_architecture

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

    @show @which interior(bottom_field)
    #@show typeof(interior(bottom_field))
    #@show size(interior(bottom_field))
    #@show typeof(bottom_field)
    #@show size(bottom_field)
    @show @which interior(bottom_field.data, location(bottom_field), bottom_field.grid, bottom_field.indices)
    @show @which interior(bottom_field.data, location(bottom_field), topology(bottom_field.grid), size(bottom_field.grid), halo_size(bottom_field.grid), bottom_field.indices)

    @show @which interior(ib.bottom_height)
    #@show typeof(interior(ib.bottom_height))
    #@show size(interior(ib.bottom_height))
    #@show typeof(ib.bottom_height)
    #@show size(ib.bottom_height)
    @show @which interior(ib.bottom_height.data, location(ib.bottom_height), ib.bottom_height.grid, ib.bottom_height.indices)
    @show @which interior(ib.bottom_height.data, location(ib.bottom_height), topology(ib.bottom_height.grid), size(ib.bottom_height.grid), halo_size(ib.bottom_height.grid), ib.bottom_height.indices)

    #my_set_to_field!(bottom_field, ib.bottom_height)

    interior(bottom_field.data, location(bottom_field), topology(bottom_field.grid), size(bottom_field.grid), halo_size(bottom_field.grid), bottom_field.indices) .= interior(ib.bottom_height.data, location(ib.bottom_height), topology(ib.bottom_height.grid), size(ib.bottom_height.grid), halo_size(ib.bottom_height.grid), ib.bottom_height.indices)
    return bottom_field
end


function my_set_to_field!(u, v)
    # We implement some niceities in here that attempt to copy halo data,
    # and revert to copying just interior points if that fails.

    if child_architecture(u) === child_architecture(v)
        # Note: we could try to copy first halo point even when halo
        # regions are a different size. That's a bit more complicated than
        # the below so we leave it for the future.

        try # to copy halo regions along with interior data
            parent(u) .= parent(v)
        catch # this could fail if the halo regions are different sizes?
            # copy just the interior data
            @info "Halo regions are different sizes, got here"
            interior(u) .= interior(v)
        end
    else
        v_data = on_architecture(child_architecture(u), v.data)

        # As above, we permit ourselves a little ambition and try to copy halo data:
        try
            parent(u) .= parent(v_data)
        catch
            interior(u) .= interior(v_data, location(v), v.grid, v.indices)
        end
    end

    return u
end

@info "Generating model..."
grid = my_data_free_ocean_climate_model_init(ReactantState())

@allowscalar underlying_grid = my_materialize_free_surface(grid)

materialized_ib = my_materialize_immersed_boundary(underlying_grid, grid.immersed_boundary)

@info "Done!"