zonal_wind(λ, φ) = 4 * sind(2φ)^2 - 2 * exp(-(abs(φ) - 12)^2 / 72)
sunlight(λ, φ) = -200 - 600 * cosd(φ)^2
Tatm(λ, φ, z=0) = 30 * cosd(φ)

function set_tracers(T, Ta, u, ua, shortwave, Qs)
    T .= Ta .+ 273.15
    u .= ua
    shortwave .= Qs
    nothing
end

function data_free_ocean_climate_model_init(
    arch::Architectures.AbstractArchitecture=Architectures.ReactantState();
    resolution::Real = 2, # 1/4 for quarter degree
    Nz::Int = 20, # eventually we want to increase this to between 100-600
    )
    Nx, Ny = resolution_to_points(resolution)
    return data_free_ocean_climate_model_init(arch, Nx, Ny, Nz)
end

function data_free_ocean_climate_model_init(
    arch::Architectures.AbstractArchitecture,
    Nx::Int, Ny::Int, Nz::Int;
    Δt = 30,
    halo = (8, 8, 8),
    grid_type = :gaussian_islands,
    set_initial_conditions = true,
    free_surface = SplitExplicitFreeSurface(substeps=30),
    )

    grid = if grid_type === :gaussian_islands
        gaussian_islands_tripolar_grid(arch, Nx, Ny, Nz; halo)
    elseif grid_type === :simple_tripolar
        simple_tripolar_grid(arch, Nx, Ny, Nz; halo)
    else
        error("grid_type=$grid_type must be :gaussian_islands or :simple_tripolar.")
    end

    # See visualize_ocean_climate_simulation.jl for information about how to
    # visualize the results of this run.
    ocean = @gbprofile "ocean_simulation" ocean_simulation(grid; free_surface, Δt)

    if set_initial_conditions
        @gbprofile "set_ocean_model" set!(ocean.model, T=Tᵢ, S=Sᵢ)
    end

    # Set up an atmosphere
    atmos_times = range(0, 1days, length=24)

    atmos_grid = LatitudeLongitudeGrid(arch,
                                       size = (360, 180),
                                       longitude = (0, 360),
                                       latitude = (-90, 90),
                                       topology = (Periodic, Bounded, Flat))

    atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times)

    if set_initial_conditions
        Ta = Field{Center, Center, Nothing}(atmos_grid)
        ua = Field{Center, Center, Nothing}(atmos_grid)
        Qs = Field{Center, Center, Nothing}(atmos_grid)

        set!(Ta, Tatm)
        set!(ua, zonal_wind)
        set!(Qs, sunlight)

        if parent(atmosphere.tracers.T) isa Reactant.ConcreteRArray
            if Reactant.precompiling()
                @code_hlo set_tracers(parent(atmosphere.tracers.T), parent(Ta), parent(atmosphere.velocities.u), parent(ua), parent(atmosphere.downwelling_radiation.shortwave), parent(Qs))
            else
                @jit set_tracers(parent(atmosphere.tracers.T), parent(Ta), parent(atmosphere.velocities.u), parent(ua), parent(atmosphere.downwelling_radiation.shortwave), parent(Qs))
            end
        else
            set_tracers(parent(atmosphere.tracers.T), parent(Ta), parent(atmosphere.velocities.u), parent(ua), parent(atmosphere.downwelling_radiation.shortwave), parent(Qs))
        end
    end

    parent(atmosphere.tracers.q) .= 0

    # Atmospheric model
    radiation = Radiation(arch)

    # Coupled model
    solver_stop_criteria = FixedIterations(5) # note: more iterations = more accurate
    atmosphere_ocean_flux_formulation = SimilarityTheoryFluxes(; solver_stop_criteria)
    interfaces = ComponentInterfaces(atmosphere, ocean; radiation, atmosphere_ocean_flux_formulation)
    coupled_model = @gbprofile "OceanSeaIceModel" OceanSeaIceModel(ocean; atmosphere, radiation, interfaces)

    return coupled_model
end # data_free_ocean_climate_model_init
