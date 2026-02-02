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
using Oceananigans.Utils: launch!
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

    atmosphere_tracers = (T = atmosphere.tracers.T.data,
                          q = atmosphere.tracers.q.data)

    Qs = atmosphere.downwelling_radiation.shortwave
    Qℓ = atmosphere.downwelling_radiation.longwave
    downwelling_radiation = (shortwave=Qs.data, longwave=Qℓ.data)
    freshwater_flux = map(ϕ -> ϕ.data, atmosphere.freshwater_flux)
    atmosphere_pressure = atmosphere.pressure.data

    # Extract info for time-interpolation
    u = atmosphere.velocities.u # for example
    atmosphere_times = u.times
    atmosphere_backend = u.backend
    atmosphere_time_indexing = u.time_indexing

    atmosphere_fields = exchanger.state
    space_fractional_indices = exchanger.regridder

    # Simplify NamedTuple to reduce parameter space consumption.
    # See https://github.com/CliMA/ClimaOcean.jl/issues/116.
    atmosphere_data = (u = atmosphere_fields.u.data,
                       v = atmosphere_fields.v.data,
                       T = atmosphere_fields.T.data,
                       p = atmosphere_fields.p.data,
                       q = atmosphere_fields.q.data,
                       Qs = atmosphere_fields.Qs.data,
                       Qℓ = atmosphere_fields.Qℓ.data,
                       Mp = atmosphere_fields.Mp.data)

    kernel_parameters = interface_kernel_parameters(grid)

    # Assumption, should be generalized
    ua = atmosphere.velocities.u

    times = ua.times
    time_indexing = ua.time_indexing
    t = clock.time
    time_interpolator = TimeInterpolator(ua.time_indexing, times, clock.time)

    launch!(arch, grid, kernel_parameters,
            _my_interpolate_primary_atmospheric_state!,
            atmosphere_data,
            space_fractional_indices,
            time_interpolator,
            grid,
            atmosphere_velocities,
            atmosphere_tracers,
            atmosphere_pressure,
            downwelling_radiation,
            freshwater_flux,
            atmosphere_backend,
            atmosphere_time_indexing)

end

@kernel function _my_interpolate_primary_atmospheric_state!(surface_atmos_state,
                                                         space_fractional_indices,
                                                         time_interpolator,
                                                         exchange_grid,
                                                         atmos_velocities,
                                                         atmos_tracers,
                                                         atmos_pressure,
                                                         downwelling_radiation,
                                                         prescribed_freshwater_flux,
                                                         atmos_backend,
                                                         atmos_time_indexing)

    i, j = @index(Global, NTuple)

    ii = space_fractional_indices.i
    jj = space_fractional_indices.j
    fi = get_fractional_index(i, j, ii)
    fj = get_fractional_index(i, j, jj)

    x_itp = FractionalIndices(fi, fj, nothing)
    t_itp = time_interpolator
    atmos_args = (x_itp, t_itp, atmos_backend, atmos_time_indexing)

    uₐ = interp_atmos_time_series(atmos_velocities.u, atmos_args...)
    vₐ = interp_atmos_time_series(atmos_velocities.v, atmos_args...)
    Tₐ = interp_atmos_time_series(atmos_tracers.T,    atmos_args...)
    qₐ = interp_atmos_time_series(atmos_tracers.q,    atmos_args...)
    pₐ = interp_atmos_time_series(atmos_pressure,     atmos_args...)

    Qs = interp_atmos_time_series(downwelling_radiation.shortwave, atmos_args...)
    Qℓ = interp_atmos_time_series(downwelling_radiation.longwave,  atmos_args...)

    # Usually precipitation
    Mh = interp_atmos_time_series(prescribed_freshwater_flux, atmos_args...)

    # Convert atmosphere velocities (usually defined on a latitude-longitude grid) to
    # the frame of reference of the native grid
    kᴺ = size(exchange_grid, 3) # index of the top ocean cell
    uₐ, vₐ = intrinsic_vector(i, j, kᴺ, exchange_grid, uₐ, vₐ)

    @inbounds begin
        surface_atmos_state.u[i, j, 1] = uₐ
        surface_atmos_state.v[i, j, 1] = vₐ
        surface_atmos_state.T[i, j, 1] = Tₐ
        surface_atmos_state.p[i, j, 1] = pₐ
        surface_atmos_state.q[i, j, 1] = qₐ
        surface_atmos_state.Qs[i, j, 1] = Qs
        surface_atmos_state.Qℓ[i, j, 1] = Qℓ
        surface_atmos_state.Mp[i, j, 1] = Mh
    end
end


@info "Compiling..."

atmosphere = coupled_model.atmosphere
exchanger  = coupled_model.interfaces.exchanger
grid       = exchanger.grid

rfirst! = @compile raise=true sync=true my_interpolate_state!(exchanger.atmosphere, grid, atmosphere, coupled_model.clock)

@info "Done!"
