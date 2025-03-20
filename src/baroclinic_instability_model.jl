using Oceananigans
using Reactant
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf

@inline function initial_buoyancy(λ, φ, z, p)
    #γ = π/2 - 2π * (p.φ₀ - φ) / p.Δφ
    γ = π/2 + 2π * λ / p.Δφ
    b = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return p.N² * z + p.Δb * b
end

function baroclinic_instability_model_init(arch; resolution, Δt)

    Lz = 1kilometers     # depth [m]
    Ny = Base.Int(20 / resolution)
    Nz = 50
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    closure = VerticalScalarDiffusivity(κ=1e-5, ν=1e-4)

    grid = LatitudeLongitudeGrid(arch,
                                 #topology = (Periodic, Bounded, Bounded),
                                 topology = (Bounded, Bounded, Bounded),
                                 size = (Ny, Ny, Nz),
                                 longitude = (-10, 10),
                                 latitude = (φ₀ - 10, φ₀ + 10),
                                 z = (-Lz, 0),
                                 halo = (6, 6, 6))

    model = HydrostaticFreeSurfaceModel(; grid, # closure,
                                        #coriolis = HydrostaticSphericalCoriolis(),
                                        #free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
                                        free_surface = SplitExplicitFreeSurface(substeps=1),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = :b,
                                        momentum_advection = nothing, #WENOVectorInvariant(order=5),
                                        tracer_advection = nothing) #WENO(order=5))

    # Parameters
    parameters = (; N², Δb, φ₀, Δφ = 20)

    
    ϵb = 1e-2 * Δb # noise amplitude
    Random.seed!(1234)
    bᵢ(x, y, z) = initial_buoyancy(x, y, z, parameters) #+ ϵb * randn()
    set!(model, b=bᵢ)

    model.clock.last_Δt = Δt

    return model
end

