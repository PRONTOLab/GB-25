using Oceananigans
using Reactant
using Libdl
using Reactant_jll
using Setfield

include("sharding_helpers.jl")

Reactant.Ops.DEBUG_MODE[] = true
ENV["JULIA_DEBUG"] = "Reactant_jll"
# @show Reactant_jll.cuDriverGetVersion(dlopen("libcuda.so"))

Reactant.set_default_backend("cpu") # or "gpu"

all_reactant_devices = Reactant.addressable_devices()
@assert length(all_reactant_devices) ≥ 4 "Need at least 4 Reactant devices"
reactant_devices = all_reactant_devices[1:4]

mesh = Sharding.Mesh(reshape(reactant_devices, 2, 2), (:x, :y))

# Set up a very simple Oceananigans simulation:
arch = CPU()

core_sharding = TreeSharding(Sharding.DimsSharding(mesh, (2, 2), (:x, :y)))

grid = LatitudeLongitudeGrid(
    arch,
    size=(360, 160, 100), # number of cells, can certainly increase
    halo=(7, 7, 7),
    longitude=(0, 360),
    latitude=(-80, 80),
    z=(-1000, 0)
)

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENOVectorInvariant())
@assert model.free_surface isa SplitExplicitFreeSurface

model_ra = Reactant.to_rarray(model) # without sharding
@set! model_ra.velocities.u = Reactant.to_rarray(model.velocities.u; sharding=core_sharding);
@set! model_ra.velocities.v = Reactant.to_rarray(model.velocities.v; sharding=core_sharding);
@set! model_ra.velocities.w = Reactant.to_rarray(model.velocities.w; sharding=core_sharding);

uᵢ(x, y, z) = randn()
set!(model, u=uᵢ, v=uᵢ)

simulation = Simulation(model_ra, Δt=60, stop_iteration=10, verbose=false)
pop!(simulation.callbacks, :nan_checker)

# if arch isa Oceananigans.Architectures.ReactantState
_run! = @compile sync = true run!(simulation)
# else
#     _run! = run!
# end

_run!(simulation)

@time _run!(simulation)
@time _run!(simulation)
@time _run!(simulation)
