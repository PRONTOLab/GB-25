using Oceananigans
using Reactant
using Libdl
using Reactant_jll

Reactant.Ops.DEBUG_MODE[] = true
ENV["JULIA_DEBUG"] = "Reactant_jll"
@show Reactant_jll.cuDriverGetVersion(dlopen("libcuda.so"))

arch = GPU() # CPU() to run on CPU
Nx, Ny, Nz = (360, 120, 100) # number of cells
grid = LatitudeLongitudeGrid(arch,
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-60, 60),
                             z = (-1000, 0))

# Simple model with random velocities
model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENO())
uᵢ(x, y, z) = randn()
set!(m, u=uᵢ, v=uᵢ)
r_model = Reactant.to_rarray(model)

# Deduce a stable time-step
Δx = minimum_xspacing(grid)
Δt = 0.1 / Δx

# Stop iteration for both simulations
stop_iteration = 100

simulation = Simulation(model; Δt, stop_iteration)
run!(simulation)

r_simulation = Simulation(r_model; Δt, stop_iteration)
pop!(r_simulation.callbacks, :nan_checker)

# What does this do?
# @show @code_hlo optimize=:before_kernel run!(r_simulation)

r_run! = @compile sync = true run!(r_simulation)
r_run!(r_simulation)

# Some tests
# Things ran normally:
@show iteration(r_simulation) == iteration(simulation)
@show time(r_simulation) == time(simulation)

# Data looks right:
u, v, w = model.velocities
ru, rv, rw = r_model.velocities

@show parent(u) == parent(ru)
@show parent(v) == parent(rv)
@show parent(w) == parent(rw)

#=
# Profiling
Reactant.with_profiler("./notrace4/") do
    run!(simulation)
end

Reactant.with_profiler("./retrace4/") do
    r_run!(r_simulation)
end

# Note: we may not want to use btime directly on `run!`
using BenchmarkTools

@btime run!(simulation)
@btime r_run!(r_simulation)
=#

