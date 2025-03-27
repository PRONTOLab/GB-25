using Reactant, MPI

MPI.Init()

Reactant.Distributed.initialize()

mesh=Sharding.Mesh(reshape(Reactant.devices(), 2, :), (:x, :y))

x_ra = Reactant.to_rarray(ones(4, 4); sharding=Sharding.NamedSharding(mesh, (:x, :y)))

y_ra = ConcreteRNumber(3.0);
# y_ra2 = Reactant.to_rarray(fill(3.0, (4, 4)))

# @. rad2deg(x_ra / y_ra)
res = x_ra ./ y_ra

display(res)

