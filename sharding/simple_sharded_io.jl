using Reactant
using GordonBell25: GordonBell25, local_shards_to_host, save_sharded_fields, load_global_field

GordonBell25.preamble()
GordonBell25.initialize(; single_gpu_per_process=false)

ndev = length(Reactant.devices())
@info "Devices: $ndev"

Dx, Dy = GordonBell25.factors(ndev)
mesh = Sharding.Mesh(Reactant.devices(), ("x", "y"); mesh_shape=(Dx, Dy))
sharding = Sharding.NamedSharding(mesh, ("x", "y", nothing))

truth = Float32.(reshape(1:1024, 16, 16, 4))
arr = Reactant.to_rarray(truth; sharding)

# Round-trip through shards
local_arrays, local_slices, gs = local_shards_to_host(arr)
reassembled = zeros(Float32, gs...)
for (a, s) in zip(local_arrays, local_slices)
    reassembled[s...] = a
end
@assert reassembled == truth "shard round-trip failed"
@info "local_shards_to_host OK"

# Round-trip through files
dir = mktempdir()
save_sharded_fields(dir, (; temp=arr), 0; iteration=1)
loaded = load_global_field(dir, :temp)
@assert loaded == truth "file round-trip failed"
@info "save/load OK"
