using Oceananigans
using Reactant
using Libdl
using Reactant_jll
using Random

Reactant.Ops.DEBUG_MODE[] = true
# ENV["JULIA_DEBUG"] = "Reactant_jll"
# @show Reactant_jll.cuDriverGetVersion(dlopen("libcuda.so"))

# Try to automatically guess whether we have a GPU available
arch = isempty(find_library(["libcuda.so.1", "libcuda.so"])) ? CPU() : GPU()
r_arch = Oceananigans.ReactantState()
Nx, Ny, Nz = (360, 120, 100) # number of cells

grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7),
                             longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0))
r_grid = LatitudeLongitudeGrid(r_arch, size=(Nx, Ny, Nz), halo=(7, 7, 7),
                               longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0))

FT = Float64
t = ConcreteRNumber(zero(FT))
iter = ConcreteRNumber(0)
stage = ConcreteRNumber(0)
last_Δt = ConcreteRNumber(zero(FT))
last_stage_Δt = ConcreteRNumber(zero(FT))
r_clock = Clock(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)

# One of the implest configurations we might consider:
model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENO())
r_model = HydrostaticFreeSurfaceModel(; grid=r_grid, clock=r_clock, momentum_advection=WENO())

@assert model.free_surface isa SplitExplicitFreeSurface
@assert r_model.free_surface isa SplitExplicitFreeSurface

Random.seed!(123)
uᵢ(x, y, z) = randn()
set!(model, u=uᵢ, v=uᵢ)
Random.seed!(123)
set!(r_model, u=uᵢ, v=uᵢ)

# What we normally do:
@info "--> Running simulation with stock Oceananigans"
simulation = Simulation(model, Δt=60, stop_iteration=2)
run!(simulation)

# What we want to do with Reactant:
@info "--> Running simulation with Reactant"
r_simulation = Simulation(r_model, Δt=60, stop_iteration=2)
pop!(r_simulation.callbacks, :nan_checker)
# @show @code_hlo optimize=:before_kernel run!(r_simulation)

r_run! = @compile sync = true run!(r_simulation)
r_run!(r_simulation)

@info "--> Re-running simulation with stock Oceananigans (with profiler)"
Reactant.with_profiler("./notrace4/") do
    run!(simulation)
end

@info "--> Re-running simulation with Reactant (with profiler)"
Reactant.with_profiler("./retrace4/") do
    r_run!(r_simulation)
end

using BenchmarkTools

@info "--> Benchmark stock Oceananigans"
@btime run!($simulation)
@info "--> Benchmark Reactant"
@btime r_run!($r_simulation)
