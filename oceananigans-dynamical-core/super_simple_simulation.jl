using Oceananigans
using Reactant
using Libdl
using Reactant_jll

Reactant.Ops.DEBUG_MODE[] = true
ENV["JULIA_DEBUG"] = "Reactant_jll"
# @show Reactant_jll.cuDriverGetVersion(dlopen("libcuda.so"))

Reactant.set_default_backend("cpu") # or "gpu"

# Set up a very simple Oceananigans simulation:
arch = CPU() #Oceananigans.Architectures.ReactantState() # CPU() to run on CPU

grid = LatitudeLongitudeGrid(arch,
                             size = (360, 160, 100), # number of cells, can certainly increase
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (-1000, 0))

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENOVectorInvariant())
@assert model.free_surface isa SplitExplicitFreeSurface

uᵢ(x, y, z) = randn()
set!(model, u=uᵢ, v=uᵢ)

simulation = Simulation(model, Δt=60, stop_iteration=10)
pop!(simulation.callbacks, :nan_checker)

if arch isa Oceananigans.Architectures.ReactantState
    _run! = @compile sync=true run!(simulation)
else
    _run! = run!
end

_run!(simulation)

@time _run!(simulation)
@time _run!(simulation)
@time _run!(simulation)

