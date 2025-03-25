using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using Reactant
using Random

@inline function initial_buoyancy(λ, φ, z, p)
    γ = π/2 - 2π * (p.φ₀ - φ) / p.Δφ
    b = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return p.N² * z + p.Δb * b
end

function baroclinic_instability_model(arch; resolution, Δt)

    Lz = 4kilometers # depth [m]
    Nx = Base.Int(360 / resolution)
    Ny = Base.Int(160 / resolution)
    Nz = 50
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    closure = VerticalScalarDiffusivity(κ=1e-5, ν=1e-4)
    dz = Lz / Nz
    z = collect(-Lz:dz:0)

    grid = LatitudeLongitudeGrid(arch; z,
                                 topology = (Periodic, Bounded, Bounded),
                                 size = (Ny, Ny, Nz),
                                 latitude = (-80, 80),
                                 longitude  = (0, 360),
                                 halo = (6, 6, 6))

    free_surface = SplitExplicitFreeSurface(substeps=30)
    closure = nothing #Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    model = HydrostaticFreeSurfaceModel(; grid, free_surface, closure,
        coriolis = HydrostaticSphericalCoriolis(),
        buoyancy = BuoyancyTracer(),
        tracers = (:b, :e),
        momentum_advection = WENOVectorInvariant(order=5),
        tracer_advection = WENO(order=5),
    )

    # Parameters
    parameters = (; N², Δb, φ₀, Δφ = 20)
    ϵb = 1e-2 * Δb # noise amplitude
    Random.seed!(1234)
    bᵢ(x, y, z) = initial_buoyancy(x, y, z, parameters) + ϵb * randn()
    set!(model, b=bᵢ)

    model.clock.last_Δt = Δt

    return model
end

function first_time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

function time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end

function loop!(model, Nt)
    @trace track_numbers=false for n = 1:Nt
        Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)
    end
    return nothing
end

arch = GPU()

@info "Generating model..."
r_model = baroclinic_instability_model(ReactantState(), Δt=2minutes, resolution=1/4)
c_model = baroclinic_instability_model(arch, Δt=2minutes, resolution=1/4)
GordonBell25.sync_states!(r_model, c_model)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling update_state..."
r_update_state! = @compile sync=true raise=true Oceananigans.TimeSteppers.update_state!(r_model)

@info "Compiling first time step..."
r_first_time_step! = @compile sync=true raise=true first_time_step!(r_model)

@info "Compiling time step..."
r_step! = @compile sync=true raise=true time_step!(r_model)

first_time_step!(c_model)

r_update_state!(r_model) # additionally needed for Reactant
r_first_time_step!(r_model)

@time time_step!(r_model)
@time time_step!(c_model)

@time "10 reactant model steps" begin
    for n = 1:10
        r_step!(r_model)
    end
end

@time "10 regular model steps" begin
    for n = 1:10
        time_step!(c_model)
    end
end

