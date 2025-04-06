function ocean_climate_model(arch, Nx, Ny, Nz; Δt,
    # Time step, cannot be changed after initialization
    Δt = 10seconds,

    # Date range used for 1) the initial condition at 2) the polar relaxation
    dates = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 12, 1),

    # Whether or not to restore the model state in polar regions, to correct
    # for the fact that we are running without a sea ice model.
    polar_restoring = false,
    polar_restoring_rate = 1 / 10days,
    )

    grid = earth_tripolar_grid(arch, resolution, Nz)

    if polar_restoring
        T_meta_series = ECCOMetadata(:temperature; dates, version=ECCO4Monthly())
        S_meta_series = ECCOMetadata(:salinity;    dates, version=ECCO4Monthly())
        FT = ECCORestoring(T_meta_series, grid; mask, rate=polar_restoring_rate)
        FS = ECCORestoring(S_meta_series, grid; mask, rate=polar_restoring_rate)
        forcing = (T=FT, S=FS)
    else
        forcing = NamedTuple()
    end

    free_surface = SplitExplicitFreeSurface(substeps=20)
    ocean = @gbprofile "ocean_simulation" ocean_simulation(grid; Δt, free_surface)

    T_init_meta = ClimaOcean.Metadata(:temperature; dates=first(dates), dataset=ClimaOcean.ECCO4Monthly())
    S_init_meta = ClimaOcean.Metadata(:salinity;    dates=first(dates), dataset=ClimaOcean.ECCO4Monthly())
    @gbprofile "set_ocean_model" set!(ocean.model, T=T_init_meta, S=S_init_meta)
    
    # Atmospheric model with 8 days of data at 3-hourly resolution
    atmosphere = JRA55PrescribedAtmosphere(arch, backend=JRA55NetCDFBackend(41))
    radiation = Radiation(arch)

    # Coupled model
    solver_stop_criteria = FixedIterations(20) # note: more iterations = more accurate
    atmosphere_ocean_flux_formulation = SimilarityTheoryFluxes(; solver_stop_criteria)
    interfaces = ComponentInterfaces(atmosphere, ocean; radiation, atmosphere_ocean_flux_formulation)
    coupled_model = @gbprofile "OceanSeaIceModel" OceanSeaIceModel(ocean; atmosphere, radiation, interfaces)

    coupled_model.clock.last_Δt = Δt
    ocean.model.clock.last_Δt = Δt
    atmosphere.clock.last_Δt = Δt

    return coupled_model
end # ocean_climate_model

