@inline function initial_buoyancy(λ, φ, z)
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    Δφ = 20
    γ = π/2 - 2π * (φ₀ - φ) / Δφ
    μ = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return N² * z + Δb * μ + 1e-2 * Δb * randn()
end

function baroclinic_instability_model(arch::Oceananigans.Architectures.AbstractArchitecture;
    resolution, Nz, grid_type = :simple_lat_lon, kw...)
    Nx = convert(Int, 384 / resolution)
    Ny = convert(Int, 192 / resolution)
    return baroclinic_instability_model(arch::Oceananigans.Architectures.AbstractArchitecture, Nx, Ny, Nz; grid_type=:simple_lat_lon, kw...)
end

function baroclinic_instability_model(arch::Oceananigans.Architectures.AbstractArchitecture, Nx, Ny, Nz; grid_type=:simple_lat_lon, kw...)
    grid = if grid_type === :gaussian_islands
        gaussian_islands_tripolar_grid(arch, Nx, Ny, Nz)
    elseif grid_type === :simple_lat_lon
        simple_latitude_longitude_grid(arch, Nx, Ny, Nz)
    end

    return baroclinic_instability_model(grid; kw...)
end

function baroclinic_instability_model(grid::Oceananigans.Grids.AbstractGrid; Δt,
    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30),

    # TEOS10 is a 54-term polynomial that relates temperature (T),
    # and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(
        equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType)),

    # CATKE correctness is not established yet, so we are using a simpler closure
    closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(),
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
    
    model = HydrostaticFreeSurfaceModel(;
        grid, free_surface, closure, buoyancy, tracers,
        coriolis, momentum_advection, tracer_advection,
    )

    Random.seed!(42)

    if buoyancy isa SeawaterBuoyancy
        set!(model, T=Tᵢ, S=Sᵢ)
    elseif buoyancy isa BuoyancyTracer
        set!(model, b=initial_buoyancy)
    end

    model.clock.last_Δt = Δt

    return model
end

