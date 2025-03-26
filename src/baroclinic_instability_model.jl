using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using SeawaterPolynomials
using Reactant
using Random

@inline function bᵢ(λ, φ, z)
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    Δφ = 20
    γ = π/2 - 2π * (φ₀ - φ) / Δφ
    b = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return p.N² * z + Δb * b + 1e-2 * Δb * randn()
end

function baroclinic_instability_model(arch; resolution, Δt)
    Lz = 4kilometers # depth [m]
    Nx = Base.Int(360 / resolution)
    Ny = Base.Int(160 / resolution)
    Nz = 50
    dz = Lz / Nz
    z = collect(-Lz:dz:0)

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)

    #=
    grid = LatitudeLongitudeGrid(arch; z,
                                 topology = (Periodic, Bounded, Bounded),
                                 size = (Ny, Ny, Nz),
                                 latitude = (-80, 80),
                                 longitude  = (0, 360),
                                 halo = (6, 6, 6))
    =#

    free_surface = SplitExplicitFreeSurface(substeps=30)
    equation_of_state = SeawaterPolynomials.TEOS10EquationOfState() 
    buoyancy = SeawaterBuoyancy(; equation_of_state)

    # CATKE correctness is not established yet, so we are using a simpler closure
    # closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    # tracers = (:T, :S, :e)
    
    closure = VerticalScalarDiffusivity(κ=1e-5, ν=1e-4)
    tracers = (:T, :S)

    model = HydrostaticFreeSurfaceModel(;
        grid, free_surface, closure, buoyancy, tracers,
        coriolis = HydrostaticSphericalCoriolis(),
        momentum_advection = WENOVectorInvariant(order=5),
        tracer_advection = WENO(order=5),
    )

    Random.seed!(42)
    set!(model, T=Tᵢ, S=Sᵢ)

    # If using buoyancy = BuoyancyTracer()
    # set!(model, b=bᵢ)

    model.clock.last_Δt = Δt

    return model
end

