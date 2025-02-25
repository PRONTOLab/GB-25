using Oceananigans
using Reactant
using Libdl
using Reactant_jll

Reactant.Ops.DEBUG_MODE[] = true
ENV["JULIA_DEBUG"] = "Reactant_jll"
# @show Reactant_jll.cuDriverGetVersion(dlopen("libcuda.so"))

Reactant.set_default_backend("cpu") # or "gpu"

all_reactant_devices = Reactant.addressable_devices()
@assert length(all_reactant_devices) ≥ 4 "Need at least 4 Reactant devices"
reactant_devices = all_reactant_devices[1:4]

mesh = Sharding.Mesh(reshape(reactant_devices, 2, 2), (:x, :y))

# Set up a very simple Oceananigans simulation:
# arch = Oceananigans.Architectures.ReactantState() # CPU() to run on CPU
arch = Oceananigans.Architectures.ReactantState(
    Sharding.NamedSharding(mesh, (:x, :y))
)

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

simulation = Simulation(model, Δt=60, stop_iteration=10, verbose=false)
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
