using Oceananigans

# simulating multiple devices on host
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8 --xla_dump_to=xla_dumps/super_simple_simulation_sharded_2/"

using Reactant
using Libdl
using Reactant_jll
using GordonBell25
using Setfield

Reactant.Ops.DEBUG_MODE[] = true
ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

Reactant.set_default_backend("cpu") # or "gpu"

# Set up a very simple Oceananigans simulation:
arch = Oceananigans.Architectures.ReactantState() # CPU() to run on CPU

grid = LatitudeLongitudeGrid(arch,
                             size = (360, 160, 100), # number of cells, can certainly increase
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-80, 80),
                             z = (-1000, 0))

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection=WENOVectorInvariant())
@assert model.free_surface isa SplitExplicitFreeSurface

mesh = Sharding.Mesh(reshape(Reactant.devices()[1:8], 2, 2, 2), (:x, :y, :z))

u_sharded = Reactant.to_rarray(
    model.velocities.u;
    sharding=GordonBell25.TreeSharding(
        Sharding.DimsSharding(mesh, (1, 2, 3), (:x, :y, :z))
    ),
);
v_sharded = Reactant.to_rarray(
    model.velocities.v;
    sharding=GordonBell25.TreeSharding(
        Sharding.DimsSharding(mesh, (1, 2, 3), (:x, :y, :z))
    ),
);
w_sharded = Reactant.to_rarray(
    model.velocities.w;
    sharding=GordonBell25.TreeSharding(
        Sharding.DimsSharding(mesh, (1, 2, 3), (:x, :y, :z))
    ),
);

@set! model.velocities.u = u_sharded;
@set! model.velocities.v = v_sharded;
@set! model.velocities.w = w_sharded;

# uᵢ(x, y, z) = randn()
# set!(model, u=uᵢ, v=uᵢ)

simulation = Simulation(model, Δt=60, stop_iteration=10, verbose=false)
pop!(simulation.callbacks, :nan_checker)

@time if arch isa Oceananigans.Architectures.ReactantState
    _run_1! = @compile sync=true shardy_passes=:default run!(simulation);
else
    _run_1! = run!;
end

@time if arch isa Oceananigans.Architectures.ReactantState
    _run_2! = @compile sync=true shardy_passes=:to_mhlo_shardings run!(simulation);
else
    _run_2! = run!;
end

_run_1!(simulation)
_run_2!(simulation)
