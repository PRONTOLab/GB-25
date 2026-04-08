@inline function initial_buoyancy(λ, φ, z)
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    Δφ = 20
    γ = π/2 - 2π * (φ₀ - φ) / Δφ
    μ = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return N² * z + Δb * μ + 1e-2 * Δb * randn()
end


function baroclinic_instability_model(arch; resolution, Nz, kw...)
    Nx, Ny = resolution_to_points(resolution)
    return baroclinic_instability_model(arch, Nx, Ny, Nz; kw...)
end

function baroclinic_instability_model(arch, Nx, Ny, Nz; Δt,
    initial_conditions_path::Union{Nothing,String} = nothing,
    halo = (8, 8, 8),
    grid_type = :simple_lat_lon, # :gaussian_islands

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30),

    # TEOS10 is a 54-term polynomial that relates temperature (T),
    # and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(
        equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType)),

    closure = nothing,    
    # closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(),
    # closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1e-5, ν=1e-4),

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis(),

    # Simple momentum advection schemes. May need to be reconsidered
    # due to Float32.
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    )

    tracers = if buoyancy isa BuoyancyTracer
        [:b]
    elseif buoyancy isa SeawaterBuoyancy
        [:T, :S]
    else
        Symbol[]
    end

    if closure isa Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity
        push!(tracers, :e)
    elseif closure isa Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity
        push!(tracers, :e)
        push!(tracers, :ϵ)
    end

    tracers = tuple(tracers...)

    grid = if grid_type === :gaussian_islands
        gaussian_islands_tripolar_grid(arch, Nx, Ny, Nz; halo)
    elseif grid_type === :simple_lat_lon
        simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo)
    else
        error("grid_type=$grid_type must be :gaussian_islands or :simple_lat_lon.")
    end

    model = HydrostaticFreeSurfaceModel(;
        grid, free_surface, closure, buoyancy, tracers,
        coriolis, momentum_advection, tracer_advection,
    )

    Random.seed!(42)

    #=
    if buoyancy isa SeawaterBuoyancy
        set_baroclinic_instability!(model)
    elseif buoyancy isa BuoyancyTracer
        # set!(model, b=initial_buoyancy)
    end
    =#

    model.clock.last_Δt = Δt

    if initial_conditions_path !== nothing
        set_baroclinic_instability_from_file!(model, initial_conditions_path)
    end

    return model
end

function set_baroclinic_instability_from_file!(model, path::String)
    Nx_src, Ny_src, Nz_src, T_data, S_data = JLD2.jldopen(path, "r") do file
        (file["Nx"], file["Ny"], file["Nz"], file["T"], file["S"])
    end

    expected = (Nx_src, Ny_src, Nz_src)
    if size(T_data) != expected
        error("Loaded T field size $(size(T_data)) does not match (Nx, Ny, Nz) = $expected from $path")
    end
    if size(S_data) != expected
        error("Loaded S field size $(size(S_data)) does not match (Nx, Ny, Nz) = $expected from $path")
    end

    target_size = size(model.tracers.T)[1:3]
    if target_size == expected
        # Same resolution as the checkpoint. The host→device path that works
        # for both plain ReactantState and Distributed{ReactantState} is the
        # same one Oceananigans' Reactant extension uses inside
        # `set_to_function!`: build a CPU twin Field on a CPU twin of the
        # *model* grid, `set!` the host array into it (set!(::Field, ::Array)
        # works on CPU), then `copyto!(interior(target), interior(cpu_twin))`.
        # The generic `set!(::Field, ::Array)` falls through to a broadcast
        # `u .= v` which silently no-ops on a Reactant field, and
        # `copyto!(interior(reactant_field), raw_array)` likewise traces but
        # never executes.
        cpu_grid = Oceananigans.Architectures.on_architecture(Oceananigans.CPU(), model.grid)
        cpu_T = CenterField(cpu_grid)
        cpu_S = CenterField(cpu_grid)
        set!(cpu_T, T_data)
        set!(cpu_S, S_data)
        copyto!(interior(model.tracers.T), interior(cpu_T))
        copyto!(interior(model.tracers.S), interior(cpu_S))
    else
        # Different resolution: build CPU source fields and interpolate. Note
        # that Oceananigans `interpolate!` requires source and target on the
        # same architecture, so this branch only works for CPU/GPU targets.
        z_src = exponential_z_faces(; Nz=Nz_src, depth=4000, h=30)
        source_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
            size = (Nx_src, Ny_src, Nz_src),
            halo = (1, 1, 1),
            z = z_src,
            latitude = (-80, 80),
            longitude = (0, 360),
        )

        T_src = CenterField(source_grid)
        S_src = CenterField(source_grid)
        set!(T_src, T_data)
        set!(S_src, S_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(T_src)
        Oceananigans.BoundaryConditions.fill_halo_regions!(S_src)

        Oceananigans.Fields.interpolate!(model.tracers.T, T_src)
        Oceananigans.Fields.interpolate!(model.tracers.S, S_src)
    end

    return nothing
end
