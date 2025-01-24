using Oceananigans
using Oceananigans.Units: minutes, days
using Printf
using Reactant

# Reactant.Ops.DEBUG_MODE[] = true

resolution = 1/4 # degree. eg resolution=1/4 is pronounced "quarter degree"
arch = GPU() # CPU() to run on CPU
Nx = Int(360 / resolution)
Ny = Int(160 / resolution)
Nz = 100
Δt = 10minutes
# stop_time = 10Δt #30days
stop_time = 30days

grid = LatitudeLongitudeGrid(arch,
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (-1000, 0))

# A configuration for idealized global simulation:
model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = WENOVectorInvariant(),
                                    tracer_advection = WENO(order=7),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer())

@info "Built a model:"
@show model

# Baroclinically unstable initial condition:
N² = 1e-6 # vertical buoyancy gradient
db = 1e-2 * N² * 1000 / Nz
M² = 1e-7 # horizontal buoyancy gradient
φ₀ = 45
dφ = 5

step(x, a, d) = 1/2 * (tanh((x - a) / d) - 1)
bᵢ(λ, φ, z) = N² * z + M² * dφ * step(φ, φ₀, dφ) + db * randn()
set!(model, b=bᵢ)

simulation = Simulation(model; Δt, stop_time)

stopwatch = Ref(time_ns())

function progress(sim)
    model = sim.model
    u, v, w = model.velocities
    elapsed = 1e-9 * (time_ns() - stopwatch[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, maximum|u|: (%.2e, %.2e, %.2e)",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, u),
                   maximum(abs, v),
                   maximum(abs, w))
    @info msg

    stopwatch[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

run!(simulation)

#using CUDA
#CUDA.@device_code dir="cudajl" run!(simulation)

#=
# What we want to do with Reactant:
r_model = Reactant.to_rarray(model)
r_simulation = Simulation(r_model, Δt=60, stop_iteration=2)
pop!(r_simulation.callbacks, :nan_checker)
# @show @code_hlo optimize=:before_kernel run!(r_simulation)

r_run! = @compile sync = true run!(r_simulation)
r_run!(r_simulation)

Reactant.with_profiler("./notrace4/") do
    run!(simulation)
end

Reactant.with_profiler("./retrace4/") do
    r_run!(r_simulation)
end

using BenchmarkTools

@btime run!(simulation)
@btime r_run!(r_simulation)
=#

