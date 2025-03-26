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

function baroclinic_instability_model(arch; resolution, Δt,
    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30),

    # TEOS10 is a 54-term polynomial that relates temperature (T),
    # and salinity (S) to buoyancy
    equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(),
    buoyancy = SeawaterBuoyancy(; equation_of_state),

    # CATKE correctness is not established yet, so we are using a simpler closure
    # closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    closure = VerticalScalarDiffusivity(κ=1e-5, ν=1e-4),

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis(),

    # Simple momentum advection schemes. May need to be reconsidered
    # due to Float32.
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5))

    if closure isa CATKEVerticalDiffusivity
        tracers = (:T, :S, :e)
    else
        tracers = (:T, :S)
    end

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)

    model = HydrostaticFreeSurfaceModel(;
        grid, free_surface, closure, buoyancy, tracers,
        coriolis, momentum_advection, tracer_advection,
    )

    Random.seed!(42)

    if buoyancy isa SeawaterBuoyancy
        set!(model, T=Tᵢ, S=Sᵢ)
    elseif buoyancy isa BuoyancyTracer()
        set!(model, b=bᵢ)
    end

    model.clock.last_Δt = Δt

    return model
end

