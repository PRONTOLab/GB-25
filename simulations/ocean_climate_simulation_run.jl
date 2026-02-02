using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

using InteractiveUtils

using Oceananigans

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: update_state!, time_step!

using ClimaOcean.OceanSeaIceModels: interpolate_state!
using ClimaOcean.Atmospheres: _interpolate_primary_atmospheric_state!, get_fractional_index, interp_atmos_time_series
using ClimaOcean.OceanSeaIceModels.InterfaceComputations: interface_kernel_parameters

using Oceananigans.OutputReaders: TimeInterpolator
using Oceananigans.Fields: FractionalIndices
using Oceananigans.Utils: launch!, _launch!
using Oceananigans.Operators: intrinsic_vector

using KernelAbstractions: @kernel, @index

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

@info "Generating model..."
coupled_model = data_free_ocean_climate_model_init(ReactantState(); resolution=4, Nz=10)
Ninner = ConcreteRNumber(2)

GC.gc(true); GC.gc(false); GC.gc(true)

function my_interpolate_state!(exchanger, grid, atmosphere, clock)
    atmosphere_grid = atmosphere.grid

    # Basic model properties
    arch = grid.architecture

    #####
    ##### First interpolate atmosphere time series
    ##### in time and to the ocean grid.
    #####

    # We use .data here to save parameter space (unlike Field, adapt_structure for
    # fts = FieldTimeSeries does not return fts.data)
    atmosphere_velocities = (u = atmosphere.velocities.u.data,
                             v = atmosphere.velocities.v.data)

    # Extract info for time-interpolation
    u = atmosphere.velocities.u # for example
    atmosphere_times = u.times
    atmosphere_backend = u.backend
    atmosphere_time_indexing = u.time_indexing

    space_fractional_indices = exchanger.regridder

    kernel_parameters = interface_kernel_parameters(grid)

    # Assumption, should be generalized
    ua = atmosphere.velocities.u

    times = ua.times
    time_indexing = ua.time_indexing
    t = clock.time
    time_interpolator = TimeInterpolator(ua.time_indexing, times, clock.time)

    _launch!(arch, grid, kernel_parameters,
            _my_interpolate_primary_atmospheric_state!,
            space_fractional_indices,
            time_interpolator,
            grid,
            atmosphere_velocities,
            atmosphere_backend,
            atmosphere_time_indexing)

end

@kernel function _my_interpolate_primary_atmospheric_state!(space_fractional_indices,
                                                         time_interpolator,
                                                         exchange_grid,
                                                         atmos_velocities,
                                                         atmos_backend,
                                                         atmos_time_indexing)

    i, j = @index(Global, NTuple)

    # Convert atmosphere velocities (usually defined on a latitude-longitude grid) to
    # the frame of reference of the native grid
    kᴺ = size(exchange_grid, 3) # index of the top ocean cell
    uₐ, vₐ = intrinsic_vector(i, j, kᴺ, exchange_grid, 0.0, 0.0)
end


@info "Compiling..."

atmosphere = coupled_model.atmosphere
exchanger  = coupled_model.interfaces.exchanger
grid       = exchanger.grid

time_interpolator = TimeInterpolator(atmosphere.velocities.u.time_indexing, atmosphere.velocities.u.times, coupled_model.clock.time)

u = atmosphere.velocities.u # for example
atmosphere_backend = u.backend
atmosphere_time_indexing = u.time_indexing

space_fractional_indices = exchanger.atmosphere.regridder

@allowscalar fi = space_fractional_indices.i[1, 1, 1]
@allowscalar fj = space_fractional_indices.j[1, 1, 1]

x_itp = FractionalIndices(fi, fj, nothing)
t_itp = time_interpolator
atmos_args = (x_itp, t_itp, atmosphere_backend, atmosphere_time_indexing)

@show @which interp_atmos_time_series(atmosphere.velocities.u.data, atmos_args...)
@show @which interp_atmos_time_series(atmosphere.velocities.v.data, atmos_args...)

@allowscalar uₐ = interp_atmos_time_series(atmosphere.velocities.u.data, atmos_args...)
@allowscalar vₐ = interp_atmos_time_series(atmosphere.velocities.v.data, atmos_args...)

@allowscalar @show uₐ
@allowscalar @show vₐ

rfirst! = @compile raise=true sync=true my_interpolate_state!(exchanger.atmosphere, grid, atmosphere, coupled_model.clock)

@info "Done!"
