using Oceananigans
using Reactant
using Libdl
using Reactant_jll

Reactant.Ops.DEBUG_MODE[] = true
ENV["JULIA_DEBUG"] = "Reactant_jll"
@show Reactant_jll.cuDriverGetVersion(dlopen("libcuda.so"))

# arch = CPU()
# arch = Distributed(GPU(), partition=Partition(2, 2)) # distributed on 4 GPUs
arch = GPU()
Nx, Ny, Nz = (360, 120, 100) # number of cells

grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7),
                             longitude=(0, 360), latitude=(-60, 60), z=(-1000, 0))

# One of the implest configurations we might consider:
free_surface = SplitExplicitFreeSurface(substeps=30)
model = HydrostaticFreeSurfaceModel(; grid, free_surface, momentum_advection=WENO())

uᵢ(x, y, z) = randn()
set!(model, u=uᵢ, v=uᵢ)

# First form a Reactant model
r_model = Reactant.to_rarray(model)

# What we normally do:
simulation = Simulation(model, Δt=60, stop_iteration=2)
run!(simulation)

#using CUDA
#CUDA.@device_code dir="cudajl" run!(simulation)

# What we want to do with Reactant:
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

