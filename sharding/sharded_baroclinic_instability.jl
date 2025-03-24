using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf

using Reactant
#Reactant.Distributed.initialize(; single_gpu_per_process=false)

function initial_buoyancy(λ, φ, z, p)
    γ = π/2 - 2π * (p.φ₀ - φ) / p.Δφ
    b = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return p.N² * z + p.Δb * b
end

function mtn₁(λ, φ)
    λ₁ = 70
    φ₁ = 55
    dφ = 5
    return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
end

function mtn₂(λ, φ)
    λ₁ = 70
    λ₂ = λ₁ + 180
    φ₂ = 55
    dφ = 5
    return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
end



function baroclinic_instability_model(arch; resolution, Nz=100, Δt=1)

    Lz = 4kilometers # depth [m]
    Nx = Base.Int(360 / resolution)
    Ny = Base.Int(160 / resolution)
    dz = Lz / Nz
    z = -Lz:dz:0

    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z)
    gaussian_islands(λ, φ) = -Lz + (Lz + 100) * (mtn₁(λ, φ) + mtn₂(λ, φ))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands))

    #=
    grid = LatitudeLongitudeGrid(arch; z,
                                 topology = (Periodic, Bounded, Bounded),
                                 size = (Ny, Ny, Nz),
                                 longitude = (0, 360),
                                 latitude = (-80, 80),
                                 halo = (6, 6, 6))
    =#

    free_surface = SplitExplicitFreeSurface(substeps=30)
    model = HydrostaticFreeSurfaceModel(; grid, free_surface,
        coriolis = HydrostaticSphericalCoriolis(),
        buoyancy = BuoyancyTracer(),
        tracers = :b,
        momentum_advection = WENOVectorInvariant(order=5),
        tracer_advection = WENO(order=5),
    )

    # Parameters
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    parameters = (; N², Δb, φ₀, Δφ = 20)
    ϵb = 1e-2 * Δb # noise amplitude
    Random.seed!(1234)
    bᵢ(x, y, z) = initial_buoyancy(x, y, z, parameters) + ϵb * randn()
    set!(model, b=bᵢ)

    model.clock.last_Δt = Δt

    return model
end

include("../ocean-climate-simulation/common.jl")

# Ngpu_str = get(ENV, "Ngpu", "1")
# Ngpu = parse(Int, Ngpu_str)
@show Ngpu = length(Reactant.devices())

Ngpu = 1

if Ngpu == 1
    rank = 0
    arch = Oceananigans.ReactantState()
elseif Ngpu == 2
    rank = Reactant.Distributed.local_rank()

    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(2, 1, 1)
    )
else
    Rx = floor(Int, sqrt(Ngpu))
    Ry = Ngpu ÷ Rx
    rank = Reactant.Distributed.local_rank()

    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(Rx, Ry, 1)
    )
end

using Dates
@info "[$rank] Generating model..." now(UTC)
resolution_fraction_str = get(ENV, "resolution_fraction", "4")
@show resolution_fraction = parse(Float64, resolution_fraction_str)
model = baroclinic_instability_model(arch; resolution=1/resolution_fraction)

@info "[$rank] Compiling first_time_step!..." 
rfirst! = @compile first_time_step!(model)

@info "[$rank] Compiling loop..."
rstep! = @compile time_step!(model)

@time "[$rank] Running first_time_step!..." rfirst!(model)
@time "[$rank] Warming up..." rstep!(model)

rstep!(model)
rstep!(model)
rstep!(model)

@time "[$rank] Running loop..." begin
    for n = 1:10
        rstep!(model)
    end
end


