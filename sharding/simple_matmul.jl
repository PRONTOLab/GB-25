# /home/avik-pal/.julia/bin/mpiexecjl -np 4 --project=. julia --threads=32 --color=yes --startup=no GB-25/sharding/simple_sharding_problem.jl

ENV["CUDA_VISIBLE_DEVICES"] = ""
# ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using MPI
MPI.Init()  # Only needed if using MPI to detect the coordinator

using Reactant

Reactant.Distributed.initialize()

mesh = Sharding.Mesh(reshape(Reactant.devices(), 2, :), (:x, :y))

x = Reactant.to_rarray(rand(8, 8); sharding=Sharding.NamedSharding(mesh, (:x, :y)))
y = Reactant.to_rarray(rand(8, 8); sharding=Sharding.NamedSharding(mesh, (:x, :y)))

res = @jit x * y;


# if Reactant.Distributed.local_rank() == 0
    display(res)
# end
