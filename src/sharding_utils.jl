using Reactant

struct TreeSharding{S} <: Sharding.AbstractSharding
    sharding::S
end

Sharding.is_sharded(sharding::TreeSharding) = true

Sharding.ndevices(sharding::TreeSharding) = Sharding.ndevices(sharding.sharding)
Sharding.shard_type(::Type{TreeSharding{S}}, N) where {S} = Sharding.shard_type(S, N)

Base.getproperty(t::TreeSharding, x) = t
function Base.getproperty(t::TreeSharding, x::Symbol)
    x == :sharding && return getfield(t, :sharding)
    return t
end

function (sharding::TreeSharding)(
    client::Reactant.XLA.AbstractClient, device::Nothing, x::Union{AbstractArray,Number}
)
    return sharding.sharding(client, device, x)
end
