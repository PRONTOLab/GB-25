using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using Reactant

using ClimaOcean
using ClimaOcean.OceanSeaIceModels.InterfaceComputations: FixedIterations, ComponentInterfaces, CoefficientBasedFluxes, BulkTemperature, default_ao_specific_humidity, default_ai_temperature, reference_density, heat_capacity, DegreesCelsius, thermodynamics_parameters, atmosphere_ocean_interface, sea_ice_ocean_interface, atmosphere_sea_ice_interface, SeaIceSimulation, surface_flux, StateExchanger, atmosphere_exchanger
using OrthogonalSphericalShellGrids: TripolarGrid

function mtn₁(λ, φ)
    λ₁ = 70
    φ₁ = 55
    dφ = 5
    return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
end

function mtn₂(λ, φ)
    λ₁ = 70
    λ₂ = λ₁ + 180
    φ₂ = 55
    dφ = 5
    return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
end

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, resolution, Nx, Ny, Nz)
    # Time step. This must be decreased as resolution is decreased.
    Δt = 1minutes

    # Grid setup
    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    #underlying_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces,
    #                                        longitude=(0, 360), latitude=(-80, 80))
    zb = z_faces[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands);
                                active_cells_map=false)
end

function MyStateExchanger(exchange_grid, atmosphere)
    exchange_atmosphere_state = (u  = Field{Center, Center, Nothing}(exchange_grid),
                                 v  = Field{Center, Center, Nothing}(exchange_grid),
                                 T  = Field{Center, Center, Nothing}(exchange_grid),
                                 q  = Field{Center, Center, Nothing}(exchange_grid),
                                 p  = Field{Center, Center, Nothing}(exchange_grid),
                                 Qs = Field{Center, Center, Nothing}(exchange_grid),
                                 Qℓ = Field{Center, Center, Nothing}(exchange_grid),
                                 Mp = Field{Center, Center, Nothing}(exchange_grid))

    exchanger = atmosphere_exchanger(atmosphere, exchange_grid)

    return StateExchanger(exchange_grid, exchange_atmosphere_state, exchanger)
end

function data_free_ocean_climate_simulation_init(
    arch::Architectures.AbstractArchitecture=Architectures.ReactantState();
    # Horizontal resolution
    resolution::Real = 2, # 1/4 for quarter degree
    Nx::Int = convert(Int, 360 / resolution),
    Ny::Int = convert(Int, 170 / resolution),
    # Vertical resolution
    Nz::Int = 20, # eventually we want to increase this to between 100-600
    )

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nx, Ny, Nz)
    atmos_times = range(0, 1days, length=24)
    atmos_grid = LatitudeLongitudeGrid(arch,
                                       size = (360, 180),
                                       longitude = (0, 360),
                                       latitude = (-90, 90),
                                       topology = (Periodic, Bounded, Flat))

    atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times)
    MyStateExchanger(grid, atmosphere)
end # data_free_ocean_climate_simulation_init
