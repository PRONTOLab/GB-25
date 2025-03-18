# This file implements data_free_ocean_model_init

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

# Simple initial condition for producing pretty pictures
function smooth_step(φ)
    φ₀ = 40
    dφ = 5
    return (1 - tanh((abs(φ) - φ₀) / dφ)) / 2
end

function Tᵢ(λ, φ, z)
    dTdz = 1e-3
    return (30 + dTdz * z) * smooth_step(φ) + rand()
end

function Sᵢ(λ, φ, z)
    dSdz = - 5e-3
    return dSdz * z + rand()
end

zonal_wind(λ, φ) = 4 * sind(2φ)^2 - 2 * exp(-(abs(φ) - 12)^2 / 72)
sunlight(λ, φ) = -200 - 600 * cosd(φ)^2
Tatm(λ, φ, z=0) = 30 * cosd(φ)

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, resolution, Nz)
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 180 / resolution)

    # Grid setup
    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    zb = z_faces[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

    return @gbprofile "ImmersedBoundaryGrid" ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands);
                                                                  active_cells_map=false)
end

function set_tracers(T, Ta, u, ua, shortwave, Qs)
    T .= Ta .+ 273.15
    u .= ua
    shortwave .= Qs
    nothing
end

function data_free_ocean_model_init(
    arch::Architectures.AbstractArchitecture=Architectures.ReactantState();
    # Horizontal resolution
    resolution::Real = 2, # 1/4 for quarter degree
    # Vertical resolution
    Nz::Int = 20, # eventually we want to increase this to between 100-600
    )

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)

    Δt = 30seconds
    free_surface = ClimaOcean.OceanSimulations.default_free_surface(grid, fixed_Δt=Δt)
    ocean = @gbprofile "ocean_simulation" ocean_simulation(grid; Δt, free_surface)
    @gbprofile "set_ocean_model" set!(ocean.model, T=Tᵢ, S=Sᵢ)

    # Set up an atmosphere
    atmos_times = range(0, 1days, length=24)

    atmos_grid = LatitudeLongitudeGrid(arch,
                                       size = (360, 180),
                                       longitude = (0, 360),
                                       latitude = (-90, 90),
                                       topology = (Periodic, Bounded, Flat))

    atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times)

    Ta = Field{Center, Center, Nothing}(atmos_grid)
    ua = Field{Center, Center, Nothing}(atmos_grid)
    Qs = Field{Center, Center, Nothing}(atmos_grid)

    set!(Ta, Tatm)
    set!(ua, zonal_wind)
    set!(Qs, sunlight)

    if arch isa Architectures.ReactantState
        if Reactant.precompiling()
            @code_hlo set_tracers(parent(atmosphere.tracers.T), parent(Ta), parent(atmosphere.velocities.u), parent(ua), parent(atmosphere.downwelling_radiation.shortwave), parent(Qs))
        else
            @jit set_tracers(parent(atmosphere.tracers.T), parent(Ta), parent(atmosphere.velocities.u), parent(ua), parent(atmosphere.downwelling_radiation.shortwave), parent(Qs))
        end
    else
        set_tracers(parent(atmosphere.tracers.T), parent(Ta), parent(atmosphere.velocities.u), parent(ua), parent(atmosphere.downwelling_radiation.shortwave), parent(Qs))
    end

    parent(atmosphere.tracers.q) .= 0

    # Atmospheric model
    radiation = Radiation(arch)

    # Coupled model
    solver_stop_criteria = FixedIterations(5) # note: more iterations = more accurate
    atmosphere_ocean_flux_formulation = SimilarityTheoryFluxes(; solver_stop_criteria)
    interfaces = ComponentInterfaces(atmosphere, ocean; radiation, atmosphere_ocean_flux_formulation)
    coupled_model = @gbprofile "OceanSeaIceModel" OceanSeaIceModel(ocean; atmosphere, radiation, interfaces)

    coupled_model.clock.last_Δt = Δt
    ocean.model.clock.last_Δt = Δt
    atmosphere.clock.last_Δt = Δt

    return coupled_model
end # data_free_ocean_model_init

