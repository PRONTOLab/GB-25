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
    initial_conditions_path::Union{Nothing,String} = joinpath(dirname(@__DIR__), "simulations", "initial_conditions", "baroclinic_ic_quarter_degree.jld2"),
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

    # Workflow:
    #   1. T_data, S_data are already loaded as CPU host arrays.
    #   2. Build source Fields on the *model's* architecture (CPU/GPU/
    #      ReactantState/Distributed) using a source grid sized to the
    #      checkpoint, and copy the host data into them via a CPU twin —
    #      this is the only host→device write pattern that works for
    #      `ReactantField` (the generic `set!(::Field, ::Array)` falls
    #      through to a broadcast that silently no-ops on Reactant).
    #   3. `interpolate!` from the on-arch source Fields onto
    #      `model.tracers.T/S`. Source and target live on the same
    #      architecture, so Oceananigans' Reactant extension dispatches
    #      to `_set_to_field!` / the device-side interpolate path.
    arch = model.grid.architecture

    # Uniform z so that the source grid is `ZRegularLLG` and
    # `fractional_z_index` is constant-time (no binary search). The
    # binary-search path uses a dynamic-trip-count `while` loop that
    # Reactant's MLIR pass manager currently fails to lower under sharding.
    source_grid = LatitudeLongitudeGrid(arch;
        size = (Nx_src, Ny_src, Nz_src),
        halo = (1, 1, 1),
        z = (-4000, 0),
        latitude = (-80, 80),
        longitude = (0, 360),
    )

    # Reactant's set!(::ReactantField, ::Array) silently no-ops, so for Reactant
    # archs we go through a CPU twin and copyto! the interiors. For plain CPU/CUDA
    # archs the generic set!(::Field, ::Array) handles host→device transfer
    # directly, and the copyto!(SubArray{CuArray}, SubArray{Array}) path falls
    # into a scalar GPU index loop, so we use set! instead.
    is_reactant = arch isa Oceananigans.Architectures.ReactantState ||
                  (arch isa Oceananigans.DistributedComputations.Distributed &&
                   Oceananigans.Architectures.child_architecture(arch) isa Oceananigans.Architectures.ReactantState)

    T_src = CenterField(source_grid)
    S_src = CenterField(source_grid)

    if is_reactant
        cpu_source_grid = Oceananigans.Architectures.on_architecture(Oceananigans.CPU(), source_grid)
        cpu_T_src = CenterField(cpu_source_grid)
        cpu_S_src = CenterField(cpu_source_grid)
        set!(cpu_T_src, T_data)
        set!(cpu_S_src, S_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(cpu_T_src)
        Oceananigans.BoundaryConditions.fill_halo_regions!(cpu_S_src)
        copyto!(interior(T_src), interior(cpu_T_src))
        copyto!(interior(S_src), interior(cpu_S_src))
    else
        set!(T_src, T_data)
        set!(S_src, S_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(T_src)
        Oceananigans.BoundaryConditions.fill_halo_regions!(S_src)
    end

    Oceananigans.Fields.interpolate!(model.tracers.T, T_src)
    Oceananigans.Fields.interpolate!(model.tracers.S, S_src)

    return nothing
end
