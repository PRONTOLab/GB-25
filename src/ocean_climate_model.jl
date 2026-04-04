const GB25_ARTIFACTS_URL = "https://github.com/glwagner/GB25Artifacts/releases/download/v1.0"

function download_artifact(filename)
    url = GB25_ARTIFACTS_URL * "/" * filename
    @info "Downloading $filename from $url..."
    Downloads.download(url, filename)
    @info "Downloaded $filename."
    return filename
end

function ensure_artifact(filename)
    if !isfile(filename)
        download_artifact(filename)
    end
    return filename
end

function ocean_climate_model_init(
    arch::Architectures.AbstractArchitecture = CPU();
    resolution::Real = 1/6,
    Nz::Int = 20,
    bathymetry_file::String = "bathymetry_sixth_degree.jld2",
    initial_conditions_file::String = "ecco2_initial_conditions_sixth_degree.jld2",
    )

    ensure_artifact(bathymetry_file)
    ensure_artifact(initial_conditions_file)

    # Build grid
    grid = simple_latitude_longitude_grid(arch, resolution, Nz)

    # Load and interpolate bathymetry
    bottom_height_data = jldopen(bathymetry_file) do file
        file["bottom_height"]
    end

    Nx_bathy, Ny_bathy = size(bottom_height_data)
    topo = (Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded, Oceananigans.Grids.Flat)
    bathy_grid = LatitudeLongitudeGrid(arch; topology=topo,
                                       size = (Nx_bathy, Ny_bathy),
                                       longitude = (0, 360),
                                       latitude = (-80, 80))

    source_field = Field{Center, Center, Nothing}(bathy_grid)
    set!(source_field, bottom_height_data)
    fill_halo_regions!(source_field)

    bottom_height = Field{Center, Center, Nothing}(grid)
    if arch isa Architectures.ReactantState
        @jit interpolate!(bottom_height, source_field)
    else
        interpolate!(bottom_height, source_field)
    end
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

    # Build ocean simulation
    Δt = 30seconds
    free_surface = SplitExplicitFreeSurface(substeps=30)
    ocean = ocean_simulation(grid; free_surface, Δt)

    # Load and interpolate ECCO2 initial conditions
    ic = jldopen(initial_conditions_file)
    T_cached = ic["T"]
    S_cached = ic["S"]
    Nx_ic    = ic["Nx"]
    Ny_ic    = ic["Ny"]
    Nz_ic    = ic["Nz"]
    close(ic)

    ic_z = exponential_z_faces(; Nz=Nz_ic, depth=4000, h=30)
    ic_grid = LatitudeLongitudeGrid(arch; size=(Nx_ic, Ny_ic, Nz_ic), halo=(8, 8, 8),
                                    z=ic_z, latitude=(-80, 80), longitude=(0, 360))

    Nx_target, Ny_target = resolution_to_points(resolution)
    z_target = exponential_z_faces(; Nz, depth=4000, h=30)
    target_grid = LatitudeLongitudeGrid(arch; size=(Nx_target, Ny_target, Nz), halo=(8, 8, 8),
                                        z=z_target, latitude=(-80, 80), longitude=(0, 360))

    src_T = Field{Center, Center, Center}(ic_grid); set!(src_T, T_cached); fill_halo_regions!(src_T)
    src_S = Field{Center, Center, Center}(ic_grid); set!(src_S, S_cached); fill_halo_regions!(src_S)

    dst_T = Field{Center, Center, Center}(target_grid)
    dst_S = Field{Center, Center, Center}(target_grid)

    if arch isa Architectures.ReactantState
        @jit interpolate!(dst_T, src_T)
        @jit interpolate!(dst_S, src_S)
    else
        interpolate!(dst_T, src_T)
        interpolate!(dst_S, src_S)
    end

    set!(ocean.model, T=Array(interior(dst_T)), S=Array(interior(dst_S)))

    # Set up atmosphere
    atmos_times = range(0, 1days, length=24)

    atmos_topo = (Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded, Oceananigans.Grids.Flat)
    atmos_grid = LatitudeLongitudeGrid(arch; topology=atmos_topo,
                                       size = (360, 180),
                                       longitude = (0, 360),
                                       latitude = (-90, 90))

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

    radiation = Radiation(arch)

    solver_stop_criteria = FixedIterations(5)
    atmosphere_ocean_flux_formulation = SimilarityTheoryFluxes(; solver_stop_criteria)
    interfaces = ComponentInterfaces(atmosphere, ocean; radiation, atmosphere_ocean_flux_formulation)
    coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation, interfaces)

    return coupled_model
end
