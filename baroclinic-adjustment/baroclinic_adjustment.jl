using Oceananigans
using Reactant
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using JSON

#=
params = open(JSON.parse, joinpath(@__DIR__, "params.json"))
in_precis = eval(Meta.parse(params["precis"]))
in_arch = eval(Meta.parse(params["arch"]))
in_resol = eval(Meta.parse(params["resol"]))
=#

# Set default floating point type
FT = Float32 #in_precis
Oceananigans.defaults.FloatType = FT

# Architecture
arch = CPU() #ReactantState() #in_arch
resolution = 1/4 #in_resol
Lz = 1kilometers     # depth [m]
Ny = Base.Int(20 / resolution)
Nz = 50
N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
Δb = 0.005 # [m/s²] buoyancy difference
φ₀ = 50
closure = VerticalScalarDiffusivity(FT; κ=1e-5, ν=1e-4)
prefix = joinpath(@__DIR__, "baroclinic_adjustment_$FT")
stop_time = 800days

# @info "Nx, Ny, Nz = $Ny, $Ny, $Nz"

grid = LatitudeLongitudeGrid(arch,
                             topology = (Periodic, Bounded, Bounded),
                             size = (Ny, Ny, Nz),
                             longitude = (-10, 10),
                             latitude = (φ₀ - 10, φ₀ + 10),
                             z = (-Lz, 0),
                             halo = (6, 6, 6))

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENOVectorInvariant(),
                                    tracer_advection = WENO(order=7))

@info model

# Parameters
parameters = (; N², Δb, φ₀, Δφ = 20)

@inline function bᵢ(λ, φ, z, p)
    γ = π/2 - 2π * (p.φ₀ - φ) / p.Δφ
    b = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return p.N² * z + p.Δb * b
end

ϵb = 1e-2 * Δb # noise amplitude
Random.seed!(1234)
bᵢ(x, y, z) = bᵢ(x, y, z, parameters) + ϵb * randn()
set!(model, b=bᵢ)

# time step
dx = minimum_xspacing(grid)
Δt = 0.1 * dx / 2 # c * dx / max(U)

# simulation = Simulation(model; Δt, stop_iteration=100)
#simulation = Simulation(model; Δt, stop_time)

stop_iteration = ceil(Int, stop_time / Δt)
simulation = Simulation(model; Δt, stop_iteration)

wall_clock = Ref(time_ns())

function progress(sim)

    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   maximum(abs, sim.model.velocities.w))

    @info msg

    wall_clock[] = time_ns()

    return nothing
end

# add_callback!(simulation, progress, IterationInterval(10))

u, v, w = model.velocities
e = @at (Center, Center, Center) (u^2 + v^2) / 2
E = Average(e, dims=(1, 2, 3))
ke_ow = JLD2OutputWriter(model, (; E),
                         filename = prefix * "_kinetic_energy.jld2",
                         schedule = TimeInterval(1days),
                         overwrite_existing = true)

# simulation.output_writers[:ke] = ke_ow

Nz = size(grid, 3)
b = model.tracers.b
ζ = ∂x(v) - ∂y(u)
fields = (; u, v, w, b, ζ)
f_ow = JLD2OutputWriter(model, fields,
                        filename = prefix * "_fields.jld2",
                        indices = (:, :, Nz),
                        schedule = TimeInterval(10days),
                        overwrite_existing = true)

# simulation.output_writers[:fields] = f_ow

Ninner = 10

function inner_loop!(model)
    @trace for _ = 1:Ninner
        time_step!(model, Δt)
    end
    return nothing
end

fast_output_interval = 200
slow_output_interval = 10 * fast_output_interval

Nfast = Int(fast_output_interval / Ninner)
Nslow = Int(slow_output_interval / Nfast)

Nt = ceil(Int, stop_time / Δt)
Nouter = ceil(Int, Nt / Nslow)

@info """

    Approximate total number of iterations:   $Nt
    Number of inner iterations:               $Ninner
    "Fast output" loop over inner iterations: $Nfast ($(Nfast * Ninner))
    "Slow output" loop over inner iterations: $Nslow ($(Nslow * Nfast * Ninner))
    Outer iterations over slow output:        $Nouter ($(Nouter * Nslow * Nfast * Ninner))
"""

if arch isa ReactantState
    @time "Compiling first time step" begin
        compiled_first_time_step! = @compile time_step!(model, Δt, euler=true)
    end

    @time "Compiling inner loop" begin
        compiled_inner_loop! = @compile inner_loop!(model)
    end

else
    compiled_first_time_step!(model, Δt) = time_step!(model, Δt, euler=true)
    compiled_inner_loop!(model) = inner_loop!(model)
end

using Oceananigans.OutputWriters: write_output!

@time "Running $Nt time steps..." begin
    if iteration(simulation) == 1
        compiled_first_time_step!(model, Δt)
    end

    for outer = 1:Nouter
        for slow = 1:Nslow
            for fast = 1:Nfast
                compiled_inner_loop!(model)
                progress(simulation)
            end
            @time "Writing fast output..." write_output!(ke_ow, simulation.model)
        end
        @time "Writing slow output..." write_output!(f_ow, simulation.model)
    end
end

