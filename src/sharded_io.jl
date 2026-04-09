using Reactant
using Oceananigans
using Serialization

"""
    local_shards_to_host(arr)

Extract the addressable (local) shards to host memory.

For `Reactant.ConcreteIFRTArray`: extracts only the local device shards
**without** any cross-process all-gather.

For plain `AbstractArray` (serial/CPU): wraps the array as a single shard.

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

function local_shards_to_host(arr::AbstractArray{T,N}) where {T,N}
    host = Array{T,N}(arr)
    all_slices = Tuple(1:s for s in size(arr))
    return [host], [all_slices], size(arr)
end

"""
    resolve_z_indices(nz, spec; halo=0)

Convert a z-level specification into concrete indices into the full data array
(which may include `halo` cells on each side).

Symbolic names refer to **interior** positions:
- `:bottom`      → `halo + 1`
- `:middle`/`:mid` → `halo + interior_nz ÷ 2`
- `:top`         → `halo + interior_nz`  (i.e. `nz - halo`)

Integer specs are treated as interior-relative and shifted by `halo`.
"""
function resolve_z_indices(nz::Int, spec; halo::Int=0)
    nz_interior = nz - 2 * halo
    spec === nothing && return collect(1:nz)
    if eltype(spec) <: Integer
        all(1 .<= spec .<= nz_interior) || error("z_indices out of range [1, $nz_interior]: $spec")
        return sort(unique([i + halo for i in spec]))
    elseif eltype(spec) <: Symbol
        indices = Int[]
        for s in spec
            if s === :bottom
                push!(indices, halo + 1)
            elseif s === :middle || s === :mid
                push!(indices, halo + max(1, nz_interior ÷ 2))
            elseif s === :top
                push!(indices, halo + nz_interior)
            else
                error("Unknown z level: $s. Use :bottom, :middle, or :top")
            end
        end
        return sort(unique(indices))
    end
    error("z_indices must be nothing, a vector of Ints, or a vector of Symbols")
end

"""
    save_sharded_fields(dir, fields::NamedTuple, rank::Int;
                        iteration=nothing, time=nothing, z_indices=nothing)

Save the local shards of each field to a per-rank file using Julia Serialization.
Each rank writes only its own addressable data — no cross-process communication.

`z_indices` selects which vertical levels to keep. Accepts `nothing` (all),
a vector of integer indices, or symbolic levels `[:bottom, :middle, :top]`.

Writes atomically via a temp file to avoid truncated checkpoints if the
process is killed mid-write.

Files are written as `fields_rank{R}.dat` containing a Dict with:
- `:iteration`, `:time` — simulation metadata
- `:z_indices` — which vertical levels were saved (or `nothing` for all)
- For each field name: `(local_arrays, local_slices, global_shape)`
"""
function save_sharded_fields(dir, fields::NamedTuple, rank::Int;
                             iteration=nothing, time=nothing,
                             z_indices=nothing, halo_z::Int=0)
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

        if z_indices !== nothing && length(global_shape) >= 3
            nz = global_shape[3]
            zidx = resolve_z_indices(nz, z_indices; halo=halo_z)
            local_arrays = [a[:, :, zidx] for a in local_arrays]
            nz_new = length(zidx)
            local_slices = map(local_slices) do s
                (s[1:2]..., 1:nz_new)
            end
            global_shape = (global_shape[1], global_shape[2], nz_new)
            state[:z_indices] = zidx
        end

        serializable_slices = [Tuple((first(r), last(r)) for r in s) for s in local_slices]
        state[name] = (;
            local_arrays,
            local_slices = serializable_slices,
            global_shape,
        )
    end

    tmppath = filepath * ".tmp"
    open(tmppath, "w") do io
        Serialization.serialize(io, state)
    end
    mv(tmppath, filepath; force=true)
    return filepath
end

"""
    extract_model_fields(model; field_names=nothing)

Extract field data arrays from a Breeze/Oceananigans model
as a NamedTuple suitable for `save_sharded_fields`.

`field_names` selects a subset of fields. Pass `nothing` (default) for all
fields, or a collection of `Symbol`s (e.g. `[:ρ, :ρu, :ρv, :ρw, :ρθ]`).

Uses `Oceananigans.fields(model)` which, for a Breeze `AtmosphereModel`,
includes prognostic fields, formulation fields (θ, ρθ, …), velocities,
temperature, and microphysical fields.
"""
function extract_model_fields(model; field_names=nothing)
    all_fields = Oceananigans.fields(model)
    names_to_use = field_names === nothing ? keys(all_fields) : field_names
    field_pairs = Pair{Symbol,Any}[]
    for name in names_to_use
        if hasproperty(all_fields, name)
            push!(field_pairs, name => getproperty(all_fields, name).data)
        else
            @warn "Field :$name not found in model, skipping. Available: $(keys(all_fields))"
        end
    end
    return NamedTuple(field_pairs)
end

"""
    _model_grid(model)

Extract the Oceananigans grid from a model, handling `OceanSeaIceModel` wrappers.
"""
function _model_grid(model)
    hasproperty(model, :grid) && return model.grid
    if hasproperty(model, :ocean)
        om = model.ocean
        hasproperty(om, :model) && hasproperty(om.model, :grid) && return om.model.grid
    end
    return nothing
end

"""
    save_model_state(dir, model, arch; label="checkpoint",
                     field_names=nothing, z_indices=nothing)

Save model fields to per-rank files. Automatically detects the grid's halo
size so that symbolic z_indices (`:bottom`, `:top`) refer to interior cells.

Returns the filepath on success, or `nothing` if an error occurred (logged
as a warning so the simulation is not interrupted).

- `field_names`: `nothing` for all fields, or a collection of `Symbol`s.
- `z_indices`:   `nothing` for all levels, integer indices, or symbolic
                 levels `[:bottom, :middle, :top]`.
"""
function save_model_state(dir, model, arch;
                          label="checkpoint",
                          field_names=nothing,
                          z_indices=nothing)
    rank = if arch isa Oceananigans.Distributed
        arch.local_rank
    else
        0
    end

    try
        fields = extract_model_fields(model; field_names)

        grid = _model_grid(model)
        halo_z = grid !== nothing ? Oceananigans.halo_size(grid)[3] : 0

        iter_val = try; Int(Array(model.clock.iteration)[1]); catch; nothing; end
        time_val = try; Float64(Array(model.clock.time)[1]); catch; nothing; end

        outdir = joinpath(dir, label)
        filepath = save_sharded_fields(outdir, fields, rank;
                                       iteration=iter_val, time=time_val,
                                       z_indices, halo_z)
        return filepath
    catch e
        @error "Checkpoint save failed (rank $rank), continuing simulation" exception=(e, catch_backtrace())
        return nothing
    end
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

Read iteration, time, field names, and z_indices from the first rank file found.
"""
function load_checkpoint_metadata(dir)
    rank_files = filter(f -> startswith(f, "fields_rank") && endswith(f, ".dat"), readdir(dir))
    isempty(rank_files) && error("No rank files found in $dir")
    state = open(joinpath(dir, first(sort(rank_files)))) do io
        Serialization.deserialize(io)
    end
    return (
        iteration   = state[:iteration],
        time        = state[:time],
        field_names = state[:field_names],
        z_indices   = get(state, :z_indices, nothing),
    )
end

"""
    load_all_fields(dir; halo=0)

Load every field from a checkpoint directory, optionally stripping `halo` cells
from each side of each horizontal dimension. When the checkpoint was saved with
`z_indices`, halo stripping is only applied to x and y (dims 1-2) since the
z levels are already explicitly selected.

Skips individual fields that fail to load rather than aborting entirely.

Returns `(metadata, field_data)` where
`metadata` is a NamedTuple with `:iteration`, `:time`, `:field_names`, `:z_indices`
and `field_data` is a `Dict{Symbol, Array}`.
"""
function load_all_fields(dir; halo::Int=0)
    meta = load_checkpoint_metadata(dir)
    z_was_subsetted = meta.z_indices !== nothing
    data = Dict{Symbol, Array}()
    for name in meta.field_names
        try
            raw = load_global_field(dir, name)
            if halo > 0
                interior_ranges = ntuple(ndims(raw)) do d
                    if z_was_subsetted && d >= 3
                        1:size(raw, d)
                    else
                        (halo + 1):(size(raw, d) - halo)
                    end
                end
                data[name] = raw[interior_ranges...]
            else
                data[name] = raw
            end
        catch e
            @warn "Failed to load field :$name, skipping" exception=(e, catch_backtrace())
        end
    end
    return meta, data
end
