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
    resolve_dim_indices(n, spec; halo=0)

Convert a dimension-level specification into concrete indices into the full
data array (which may include `halo` cells on each side).

Symbolic names refer to **interior** positions:
- `:bottom`/`:first`  → `halo + 1`
- `:middle`/`:mid`    → `halo + interior_n ÷ 2`
- `:top`/`:last`      → `halo + interior_n`

Integer specs are interior-relative and shifted by `halo`.
"""
function resolve_dim_indices(n::Int, spec; halo::Int=0)
    n_interior = n - 2 * halo
    spec === nothing && return collect(1:n)
    if eltype(spec) <: Integer
        all(1 .<= spec .<= n_interior) || error("indices out of range [1, $n_interior]: $spec")
        return sort(unique([i + halo for i in spec]))
    elseif eltype(spec) <: Symbol
        indices = Int[]
        for s in spec
            if s === :bottom || s === :first
                push!(indices, halo + 1)
            elseif s === :middle || s === :mid
                push!(indices, halo + max(1, n_interior ÷ 2))
            elseif s === :top || s === :last
                push!(indices, halo + n_interior)
            else
                error("Unknown level: $s. Use :bottom/:first, :middle/:mid, or :top/:last")
            end
        end
        return sort(unique(indices))
    end
    error("spec must be nothing, a vector of Ints, or a vector of Symbols")
end

const resolve_z_indices = resolve_dim_indices

"""
    slice_dim(plane::Symbol) → Int

Map a slice plane to the dimension that gets indexed (the "through" dimension):
- `:xy` → 3  (fix z, keep x-y)
- `:xz` → 2  (fix y, keep x-z)
- `:yz` → 1  (fix x, keep y-z)
"""
function slice_dim(plane::Symbol)
    plane === :xy && return 3
    plane === :xz && return 2
    plane === :yz && return 1
    error("Unknown slice plane: $plane. Use :xy, :xz, or :yz.")
end

"""
    _apply_slice(local_arrays, local_slices, global_shape, dim, idx)

Slice `local_arrays` along dimension `dim` at indices `idx`.
Returns updated `(local_arrays, local_slices, global_shape)`.
"""
function _apply_slice(local_arrays, local_slices, global_shape, dim::Int, idx::Vector{Int})
    ndim = length(global_shape)
    new_arrays = map(local_arrays) do a
        slicers = [1:size(a, d) for d in 1:ndim]
        slicers[dim] = idx
        a[slicers...]
    end
    n_new = length(idx)
    new_slices = map(local_slices) do s
        out = collect(s)
        out[dim] = 1:n_new
        Tuple(out)
    end
    new_shape = ntuple(d -> d == dim ? n_new : global_shape[d], ndim)
    return new_arrays, new_slices, new_shape
end

"""
    save_sharded_fields(dir, fields::NamedTuple, rank::Int;
                        iteration=nothing, time=nothing,
                        z_indices=nothing, halo_z::Int=0,
                        slices=nothing, halo_sizes::NTuple{3,Int}=(0,0,0))

Save the local shards of each field to a per-rank file.

Two mutually exclusive slicing modes:

**Legacy (`z_indices`)**: apply the same z-slice to every field.

**New (`slices`)**: a vector of `(field_name, plane, level_spec)` tuples
specifying per-field, per-plane slices. Each entry produces a separate
saved array keyed as `Symbol("\$(field)_\$(plane)")`:
- `plane` ∈ `{:xy, :xz, :yz}` — which 2-D cross-section to keep.
- `level_spec` — symbolic (`:bottom`, `:middle`, `:top`) or integer indices
  along the sliced-through dimension, same as the legacy `z_indices`.

Writes atomically via a temp file.
"""
function save_sharded_fields(dir, fields::NamedTuple, rank::Int;
                             iteration=nothing, time=nothing,
                             z_indices=nothing, halo_z::Int=0,
                             slices=nothing, halo_sizes::NTuple{3,Int}=(0,0,0))
    mkpath(dir)
    filepath = joinpath(dir, "fields_rank$(rank).dat")

    state = Dict{Symbol,Any}(
        :iteration => iteration,
        :time => time,
        :halo_sizes => halo_sizes,
    )

    if slices !== nothing
        saved_keys = Symbol[]
        slice_meta = []

        for (field_name, plane, level_spec) in slices
            haskey(fields, field_name) || begin
                @warn "Field :$field_name not in fields, skipping slice ($field_name, $plane, $level_spec)"
                continue
            end

            arr = Reactant.ancestor(fields[field_name])
            local_arrays, local_slices, global_shape = local_shards_to_host(arr)

            dim = slice_dim(plane)
            resolved_idx = Int[]
            if length(global_shape) >= dim
                h = halo_sizes[dim]
                n = global_shape[dim]
                resolved_idx = resolve_dim_indices(n, level_spec; halo=h)
                local_arrays, local_slices, global_shape = _apply_slice(
                    local_arrays, local_slices, global_shape, dim, resolved_idx)
            end

            key = Symbol("$(field_name)_$(plane)")
            serializable_slices = [Tuple((first(r), last(r)) for r in s) for s in local_slices]
            state[key] = (;
                local_arrays,
                local_slices = serializable_slices,
                global_shape,
            )
            push!(saved_keys, key)
            push!(slice_meta, (field=field_name, plane=plane, dim=dim, indices=resolved_idx))
        end

        state[:field_names] = saved_keys
        state[:slices] = [(; m.field, m.plane, m.dim) for m in slice_meta]

    else
        state[:field_names] = collect(keys(fields))

        for (name, field_data) in pairs(fields)
            arr = Reactant.ancestor(field_data)
            local_arrays, local_slices, global_shape = local_shards_to_host(arr)

            if z_indices !== nothing && length(global_shape) >= 3
                nz = global_shape[3]
                zidx = resolve_dim_indices(nz, z_indices; halo=halo_z)
                local_arrays, local_slices, global_shape = _apply_slice(
                    local_arrays, local_slices, global_shape, 3, zidx)
                state[:z_indices] = zidx
            end

            serializable_slices = [Tuple((first(r), last(r)) for r in s) for s in local_slices]
            state[name] = (;
                local_arrays,
                local_slices = serializable_slices,
                global_shape,
            )
        end
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
                     field_names=nothing, z_indices=nothing, slices=nothing)

Save model fields to per-rank files. Automatically detects the grid's halo
sizes so that symbolic levels (`:bottom`, `:top`, etc.) refer to interior cells.

Two slicing modes (mutually exclusive):

**Legacy** (`field_names` + `z_indices`): save the listed fields, optionally
sub-setting along z.

**New** (`slices`): a vector of `(field_name, plane, level_spec)` tuples
for arbitrary per-field cross-sections.  Example:

    slices = [
        (:T, :xy, [:bottom, :top]),   # T at z=bottom and z=top
        (:T, :xz, [:middle]),         # T at y=middle  (lon-z cross-section)
        (:w, :yz, [:middle]),         # w at x=middle  (lat-z cross-section)
    ]

Returns the filepath on success, or `nothing` on error.
"""
function save_model_state(dir, model, arch;
                          label="checkpoint",
                          field_names=nothing,
                          z_indices=nothing,
                          slices=nothing)
    rank = if arch isa Oceananigans.Distributed
        arch.local_rank
    else
        0
    end

    try
        grid = _model_grid(model)
        hs = grid !== nothing ? Oceananigans.halo_size(grid) : (0, 0, 0)
        halo_sizes = (hs[1], hs[2], hs[3])

        iter_val = try; Int(Array(model.clock.iteration)[1]); catch; nothing; end
        time_val = try; Float64(Array(model.clock.time)[1]); catch; nothing; end

        outdir = joinpath(dir, label)

        if slices !== nothing
            slice_fields = unique([s[1] for s in slices])
            fields = extract_model_fields(model; field_names=slice_fields)
            filepath = save_sharded_fields(outdir, fields, rank;
                                           iteration=iter_val, time=time_val,
                                           slices, halo_sizes)
        else
            fields = extract_model_fields(model; field_names)
            filepath = save_sharded_fields(outdir, fields, rank;
                                           iteration=iter_val, time=time_val,
                                           z_indices, halo_z=halo_sizes[3],
                                           halo_sizes)
        end
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

Read iteration, time, field names, slice metadata, and z_indices from the
first rank file found. Works with both legacy (z_indices) and new (slices)
checkpoint formats.
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
        slices      = get(state, :slices, nothing),
        halo_sizes  = get(state, :halo_sizes, nothing),
    )
end

"""
    load_all_fields(dir; halo=nothing)

Load every field/slice from a checkpoint directory, stripping halo cells from
non-sliced dimensions.

`halo` accepts:
- `nothing` (default) — reads `halo_sizes` stored in the checkpoint; if absent,
  no stripping is performed.
- `Int` — uniform halo applied to all three spatial dimensions.
- `NTuple{3,Int}` — per-dimension `(Hx, Hy, Hz)` halo sizes.

For **legacy** checkpoints (saved with `z_indices`): strips x/y halos; z is
already subsetted so left as-is.

For **new** checkpoints (saved with `slices`): strips halos on the two
retained dimensions (the sliced-through dimension is already subsetted).

Returns `(metadata, field_data)` where `field_data` is a `Dict{Symbol, Array}`.
"""
function load_all_fields(dir; halo::Union{Nothing, Int, NTuple{3,Int}}=nothing)
    meta = load_checkpoint_metadata(dir)
    slice_info = meta.slices

    hs = if halo isa NTuple{3,Int}
        halo
    elseif halo isa Int
        (halo, halo, halo)
    elseif meta.halo_sizes !== nothing
        Tuple(meta.halo_sizes)
    else
        (0, 0, 0)
    end
    strip = any(h -> h > 0, hs)

    data = Dict{Symbol, Array}()

    for name in meta.field_names
        try
            raw = load_global_field(dir, name)
            if strip
                if slice_info !== nothing
                    si = findfirst(s -> Symbol("$(s.field)_$(s.plane)") === name, slice_info)
                    sliced_dim = si !== nothing ? slice_info[si].dim : nothing
                    interior_ranges = ntuple(ndims(raw)) do d
                        h = d <= 3 ? hs[d] : 0
                        if sliced_dim !== nothing && d == sliced_dim
                            1:size(raw, d)
                        elseif h > 0
                            (h + 1):(size(raw, d) - h)
                        else
                            1:size(raw, d)
                        end
                    end
                else
                    z_was_subsetted = meta.z_indices !== nothing
                    interior_ranges = ntuple(ndims(raw)) do d
                        h = d <= 3 ? hs[d] : 0
                        if z_was_subsetted && d >= 3
                            1:size(raw, d)
                        elseif h > 0
                            (h + 1):(size(raw, d) - h)
                        else
                            1:size(raw, d)
                        end
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
