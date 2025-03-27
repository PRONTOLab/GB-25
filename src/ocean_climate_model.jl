# This file implements ocean_climate_model_init

function earth_tripolar_grid(arch::Architectures.AbstractArchitecture, resolution, Nz, zstar_vertical_coordinate)
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 180 / resolution)

    # Grid setup
    z_faces = exponential_z_faces(; Nz, depth=6000, h=30) # may need changing for very large Nz
    if zstar_vertical_coordinate
        z_faces = Oceananigans.Grids.MutableVerticalDiscretization(z_faces)
    end

    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    # Bathymetry based on ETOPO1: https://www.ncei.noaa.gov/products/etopo-global-relief-model
    bathymetry = ClimaOcean.regrid_bathymetry(underlying_grid, interpolation_passes=10, major_basins=1)

    return @gbprofile "ImmersedBoundaryGrid" ImmersedBoundaryGrid(underlying_grid,
                                                                  GridFittedBottom(bathymetry);
                                                                  active_cells_map = false)
end

function ocean_climate_model_init(
    arch::Architectures.AbstractArchitecture=Architectures.ReactantState();

    # Horizontal resolution
    resolution::Real = 2, # 1/4 for quarter degree

    # Time step, cannot be changed after initialization
    Δt = 30seconds,

    # Vertical resolution
    Nz::Int = 20, # eventually we want to increase this to between 100-600

    # Vertical coordinate
    zstar_vertical_coordinate = false,

    # Date range used for 1) the initial condition at 2) the polar relaxation
    dates = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 12, 1),

    # Use these to fiddle with the order of the advection scheme:
    momentum_advection_order = 5,
    tracer_advection_order = 5,
    vorticity_order = nothing, # if this is provided, momentum_advection_order is ignored

    # Whether or not to restore the model state in polar regions, to correct
    # for the fact that we are running without a sea ice model.
    polar_restoring = false,
    polar_restoring_rate = 1 / 10days,

    # Whether to include passive tracers, which are useful for numerics tests
    include_passive_tracers = false
    )

    grid = earth_tripolar_grid(arch, resolution, Nz, zstar_vertical_coordinate)

    if zstar_vertical_coordinate
        vertical_coordinate = Oceananigans.Models.HydrostaticFreeSurfaceModels.ZStar()
    else
	vertical_coordinate = Oceananigans.Models.HydrostaticFreeSurfaceModels.ZCoordinate()
    end

    if polar_restoring
        T_meta_series = ECCOMetadata(:temperature; dates, version=ECCO4Monthly())
        S_meta_series = ECCOMetadata(:salinity;    dates, version=ECCO4Monthly())
        FT = ECCORestoring(T_meta_series, grid; mask, rate=polar_restoring_rate)
        FS = ECCORestoring(S_meta_series, grid; mask, rate=polar_restoring_rate)
        forcing = (T=FT, S=FS)
    else
        forcing = NamedTuple()
    end

    free_surface = ClimaOcean.OceanSimulations.default_free_surface(grid, fixed_Δt=Δt)
    if isnothing(vorticity_order)
        momentum_advection = WENOVectorInvariant(; order=momentum_advection_order)
    else
        momentum_advection = WENOVectorInvariant(; vorticity_order)
    end

    tracer_advection = WENO(order=tracer_advection_order)

    if include_passive_tracers
    	tracers=(:T, :S, :e, :C_surface, :C_bottom)
    else
    	tracers=(:T, :S, :e)
    end

    ocean = @gbprofile "ocean_simulation" ocean_simulation(
        grid;
        Δt,
        free_surface,
        tracer_advection,
        momentum_advection,
        vertical_coordinate,
	tracers
    )

    T_init_meta = ClimaOcean.Metadata(:temperature; dates=first(dates), dataset=ClimaOcean.ECCO4Monthly())
    S_init_meta = ClimaOcean.Metadata(:salinity;    dates=first(dates), dataset=ClimaOcean.ECCO4Monthly())
    @gbprofile "set_ocean_model" set!(ocean.model, T=T_init_meta, S=S_init_meta)
    
    if include_passive_tracers
        # Initialize surface and bottom tracer conditions
        C_surface_init, C_bottom_init = ocean.model.tracers.C_surface, ocean.model.tracers.C_bottom
        C_surface_init_interior, C_bottom_init_interior = interior(C_surface_init), interior(C_bottom_init)
    
        # Set initial conditions to 1 at the vertical boundaries
        C_surface_init_interior[:, :, Nz] .= 1  # Uppermost level
        C_bottom_init_interior[:, :, 1] .= 1    # Lowermost level
    
        @gbprofile "set_ocean_model" set!(ocean.model, C_surface=C_surface_init, C_bottom=C_bottom_init)
    end

    # Atmospheric model with 8 days of data at 3-hourly resolution
    atmosphere = JRA55PrescribedAtmosphere(arch, backend=JRA55NetCDFBackend(41))
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
end # ocean_climate_model_init

