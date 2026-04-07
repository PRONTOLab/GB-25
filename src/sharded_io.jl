using Reactant
using Oceananigans
using Serialization

"""
    local_shards_to_host(arr::Reactant.ConcreteIFRTArray{T,N})

Extract only the addressable (local) shards of a distributed `ConcreteIFRTArray`
to host memory, **without** any cross-process all-gather.

Returns `(local_arrays, local_slices, global_shape)` where:
- `local_arrays::Vector{Array{T,N}}` — one host array per local device shard
- `local_slices::Vector` — corresponding global index ranges for each shard
- `global_shape::NTuple{N,Int}` — the full distributed array shape
"""
function local_shards_to_host(arr::Reactant.ConcreteIFRTArray{T,N}) where {T,N}
    shard_info = arr.sharding
    reactant_sharding = Sharding.unwrap_shardinfo(shard_info)
    global_shape = size(arr)

    if !Sharding.is_sharded(shard_info)
        host = Array{T,N}(undef, global_shape...)
        Reactant.XLA.to_host(arr.data, host, Sharding.NoSharding())
        all_slices = Tuple(1:s for s in global_shape)
        return [host], [all_slices], global_shape
    end

    if reactant_sharding isa Sharding.HloSharding
        (; hlo_sharding) = reactant_sharding
    else
        (; hlo_sharding) = Sharding.HloSharding(reactant_sharding, global_shape)
    end

    buffer = Reactant.XLA.synced_buffer(arr.data)
    single_device_arrays = Reactant.XLA.IFRT.disassemble_into_single_device_arrays(buffer, true)

    client = Reactant.XLA.client(buffer)
    all_devices = Reactant.XLA.get_device.((client,), reactant_sharding.mesh.device_ids)

    all_slices, _ = Reactant.XLA.sharding_to_concrete_array_indices(
        convert(Reactant.XLA.CondensedOpSharding, hlo_sharding),
        global_shape,
        reactant_sharding.mesh.logical_device_ids,
    )

    local_slices = [s for (dev, s) in zip(all_devices, all_slices) if Reactant.XLA.is_addressable(dev)]
    @assert length(local_slices) == length(single_device_arrays)

    local_arrays = map(zip(local_slices, single_device_arrays)) do (slice, dev_arr)
        shard_shape = Tuple(length(r) for r in slice)
        host_buf = Array{T}(undef, shard_shape...)
        Reactant.XLA.to_host(dev_arr, host_buf, Sharding.NoSharding())
        return host_buf
    end

    return local_arrays, local_slices, global_shape
end

"""
    save_sharded_fields(dir, fields::NamedTuple, rank::Int;
                        iteration=nothing, time=nothing)

Save the local shards of each field to a per-rank file using Julia Serialization.
Each rank writes only its own addressable data — no cross-process communication.

Files are written as `fields_rank{R}.dat` containing a Dict with:
- `:iteration`, `:time` — simulation metadata
- For each field name: `(local_arrays, local_slices, global_shape)`
"""
function save_sharded_fields(dir, fields::NamedTuple, rank::Int;
                             iteration=nothing, time=nothing)
    mkpath(dir)
    filepath = joinpath(dir, "fields_rank$(rank).dat")

    state = Dict{Symbol,Any}(
        :iteration => iteration,
        :time => time,
        :field_names => collect(keys(fields)),
    )

    for (name, field_data) in pairs(fields)
        arr = Reactant.ancestor(field_data)
        local_arrays, local_slices, global_shape = local_shards_to_host(arr)
        serializable_slices = [Tuple((first(r), last(r)) for r in s) for s in local_slices]
        state[name] = (;
            local_arrays,
            local_slices = serializable_slices,
            global_shape,
        )
    end

    open(filepath, "w") do io
        Serialization.serialize(io, state)
    end
    return filepath
end

"""
    extract_model_fields(model)

Extract all field data arrays from a Breeze/Oceananigans model
as a NamedTuple suitable for `save_sharded_fields`.

Uses `Oceananigans.fields(model)` which, for a Breeze `AtmosphereModel`,
includes prognostic fields, formulation fields (θ, ρθ, …), velocities,
temperature, and microphysical fields.
"""
function extract_model_fields(model)
    all_fields = Oceananigans.fields(model)
    field_pairs = Pair{Symbol,Any}[]
    for (name, field) in pairs(all_fields)
        push!(field_pairs, name => field.data)
    end
    return NamedTuple(field_pairs)
end

"""
    save_model_state(dir, model, arch; label="checkpoint")

Save all prognostic fields of the model to per-rank files.
"""
function save_model_state(dir, model, arch; label="checkpoint")
    rank = if arch isa Oceananigans.Distributed
        arch.local_rank
    else
        0
    end

    fields = extract_model_fields(model)

    iter_val = try; Int(Array(model.clock.iteration)[1]); catch; nothing; end
    time_val = try; Float64(Array(model.clock.time)[1]); catch; nothing; end

    outdir = joinpath(dir, label)
    filepath = save_sharded_fields(outdir, fields, rank;
                                   iteration=iter_val, time=time_val)
    return filepath
end

"""
    load_global_field(dir, field_name::Symbol; ranks=nothing)

Reconstruct a global array from per-rank shard files (offline utility).
Reads all `fields_rank*.dat` files in `dir` and assembles the named field.
"""
function load_global_field(dir, field_name::Symbol; ranks=nothing)
    rank_files = sort(filter(f -> startswith(f, "fields_rank") && endswith(f, ".dat"), readdir(dir)))

    if ranks !== nothing
        rank_files = filter(f -> any(contains(f, "rank$(r).dat") for r in ranks), rank_files)
    end

    isempty(rank_files) && error("No rank files found in $dir")

    first_state = open(joinpath(dir, first(rank_files))) do io
        Serialization.deserialize(io)
    end
    info = first_state[field_name]
    T = eltype(first(info.local_arrays))
    global_array = zeros(T, info.global_shape...)

    for rf in rank_files
        state = open(joinpath(dir, rf)) do io
            Serialization.deserialize(io)
        end
        field_info = state[field_name]
        for (arr, slice_bounds) in zip(field_info.local_arrays, field_info.local_slices)
            ranges = Tuple(a:b for (a, b) in slice_bounds)
            global_array[ranges...] = arr
        end
    end

    return global_array
end

"""
    load_checkpoint_metadata(dir)

Read iteration, time, and field names from the first rank file found.
"""
function load_checkpoint_metadata(dir)
    rank_files = filter(f -> startswith(f, "fields_rank") && endswith(f, ".dat"), readdir(dir))
    isempty(rank_files) && error("No rank files found in $dir")
    state = open(joinpath(dir, first(sort(rank_files)))) do io
        Serialization.deserialize(io)
    end
    return (iteration=state[:iteration], time=state[:time], field_names=state[:field_names])
end

