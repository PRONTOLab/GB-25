using Oceananigans
using Oceananigans.Units: minutes, days
using Reactant

# Reactant.Ops.DEBUG_MODE[] = true

resolution = 1/4 # degree. eg resolution=1/4 is pronounced "quarter degree"
arch = GPU() # CPU() to run on CPU
Nx = Int(resolution * 360)
Ny = Int(resolution * 160)
Nz = 100
Lz = 1000
Δt = 5minutes
stop_time = 10Δt #30days

grid = LatitudeLongitudeGrid(arch,
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (-Lz, 0))

# Make a bumpy bottom
function bumpy_bottom_height(λ, φ)
    h = 200 # m
    d = 2 # width of bumps in degrees
    bottom_height = -Lz
    for λ₀ in 0:10:350
        bottom_height += h * exp(-(λ - λ₀)^2 / 2dλ^2)
    end
    return bottom_height
end

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bumpy_bottom_height))

# A configuration for idealized global simulation:
model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = WENOVectorInvariant(),
                                    tracer_advection = WENO(order=7),
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer())

# Baroclinically unstable initial condition:
N² = 1e-6 # vertical buoyancy gradient
db = 1e-2 * N² * 1000 / Nz
M² = 1e-7 # horizontal buoyancy gradient
φ₀ = 45
dφ = 5

step(x, a, d) = 1/2 * (tanh((x - a) / d) - 1)
bᵢ(λ, φ, z) = N² * z + M² * step(φ, φ₀, dφ) + db * randn()
set!(model, b=bᵢ)

simulation = Simulation(model; Δt, stop_time)
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

