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
using Oceananigans.Operators: intrinsic_vector, getvalue, rotation_angle

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

    kernel_parameters = interface_kernel_parameters(grid)

    _launch!(arch, grid, kernel_parameters,
            _my_interpolate_primary_atmospheric_state!,
            grid)

end

@kernel function _my_interpolate_primary_atmospheric_state!(exchange_grid)

    i, j = @index(Global, NTuple)

    # Convert atmosphere velocities (usually defined on a latitude-longitude grid) to
    # the frame of reference of the native grid
    kᴺ = size(exchange_grid, 3) # index of the top ocean cell
    uₐ, vₐ = my_intrinsic_vector(i, j, kᴺ, exchange_grid.underlying_grid, 0.0, 0.0)
end

@inline function my_intrinsic_vector(i, j, k, grid, uₑ, vₑ)

    u = getvalue(uₑ, i, j, k, grid)
    v = getvalue(vₑ, i, j, k, grid)

    θ_degrees = rotation_angle(i, j, grid)
    sinθ = sind(θ_degrees)
    cosθ = cosd(θ_degrees)

    uᵢ = u * cosθ - v * sinθ
    vᵢ = u * sinθ + v * cosθ

    return uᵢ, vᵢ
end


@info "Compiling..."

atmosphere = coupled_model.atmosphere
exchanger  = coupled_model.interfaces.exchanger
grid       = exchanger.grid

@show @which intrinsic_vector(1, 1, size(grid, 3), grid.underlying_grid, 0.0, 0.0)

rfirst! = @compile raise=true sync=true my_interpolate_state!(exchanger.atmosphere, grid, atmosphere, coupled_model.clock)

@info "Done!"
