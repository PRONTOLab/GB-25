using Oceananigans
using Oceananigans.Units: minutes, days
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Printf
using Reactant

# Reactant.Ops.DEBUG_MODE[] = true

longitude = (0, 360)
latitude = (-80, 80)
resolution = 1/4 # degree. eg resolution=1/4 is pronounced "quarter degree"
# arch = GPU()
arch = CPU()
Nx = 360
Ny = 160
Nz = 20
Δt = 20minutes
# stop_time = 10Δt #30days
stop_time = 30days

grid = LatitudeLongitudeGrid(arch,
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (-1000, 0))

# A configuration for idealized global simulation:

# Eventually we want to use this:
# closure = CATKEVerticalDiffusivity()
# tracers = (:b, :e)

# but for now:
closure = nothing
tracers = :b

model = HydrostaticFreeSurfaceModel(; grid, closure, tracers,
                                    momentum_advection = WENOVectorInvariant(),
                                    tracer_advection = WENO(order=7),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    buoyancy = BuoyancyTracer())

@info "Built a model:"
@show model

# Baroclinically unstable initial condition:
N² = 1e-6 # vertical buoyancy gradient
M² = 1e-6 # horizontal buoyancy gradient
db = 1e-1 * M² * 1000 / Nz
φ₀ = 45
dφ = 3

step(x, a, d) = 1/2 * (tanh((x - a) / d) - 1)
bᵢ(λ, φ, z) = N² * z + M² * dφ * step(φ, φ₀, dφ) + db * randn()
set!(model, b=bᵢ)

simulation = Simulation(model; Δt, stop_time)
#simulation = Simulation(model; Δt, stop_iteration=400)

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

# To run things regularly
add_callback!(simulation, progress, IterationInterval(10))
run!(simulation)

using GLMakie

b = model.tracers.b
u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)
fig = Figure()
axb = Axis(fig[1, 1])
axz = Axis(fig[1, 2])
heatmap!(axz, view(ζ, :, :, 1))
heatmap!(axb, view(b, :, :, 1))

#=
# To run with Reactant:
r_model = Reactant.to_rarray(model)
r_simulation = Simulation(r_model; Δt, stop_time)
pop!(r_simulation.callbacks, :nan_checker)

# This may not work:
add_callback!(r_simulation, progress, IterationInterval(100))

r_run! = @compile sync = true run!(r_simulation)
r_run!(r_simulation)
=#

# Extra stuff:
#=
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

