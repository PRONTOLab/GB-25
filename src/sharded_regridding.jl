using Reactant
using Reactant: Sharding
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Fields: location, interior, regrid!
using Oceananigans.Grids: Center, Face, Periodic, Bounded, x_domain, y_domain, z_domain, topology
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: CPU, architecture

#####
##### Distributed conservative regridding (coarse -> fine) with NO host gather.
#####
##### This is the write-side counterpart of `sharded_io.jl`'s `local_shards_to_host`:
#####   * `local_shards_to_host` DISASSEMBLES a sharded array into per-shard host buffers.
#####   * the functions here BUILD per-shard host buffers (by regridding a local source
#####     subregion) and ASSEMBLE them into a sharded array.
##### The global fine field is never materialized on one device, and each process touches
##### only the source cells overlapping its own shards.
#####
##### Method: on a LatitudeLongitudeGrid the conservative area weight (∝ Δλ·Δsinφ)
##### factorizes, so Oceananigans' `regrid!` done one dimension at a time (x, then y) is
##### exact. `regrid!` is only defined for `Center`-located dimensions, so Face-located
##### fields (velocities) are handled by AVERAGING each Face dimension to a Center proxy,
##### regridding the proxy, then RECONSTRUCTING the faces on the target (½-interpolation).
#####
##### Scope: cell-centered tracers use the pure conservative path. Face-located fields
##### (u, v, U, V) go through the center-proxy round-trip: correct, but the face
##### reconstruction is interpolatory (not strictly conservative) — the standard staggered
##### treatment. Any refinement ratio works (source cell ranges are rounded outward). The
##### source and target must share the same physical domain AND the same vertical grid:
##### z is index pass-through (Center `Nz`, Face `Nz+1` like w, or 2D `Nothing`), regridded
##### level-by-level in the horizontal. Vertical refinement is not supported.

"Physical extent of a LatitudeLongitudeGrid as `((λ₁,λ₂), (φ₁,φ₂), (z₁,z₂))`."
latlon_bounds(grid) = (x_domain(grid), y_domain(grid), z_domain(grid))

# Number of Center CELLS backing a field's point count in a dimension with the given
# (location, topology). A Face location in a Bounded dimension carries the extra
# boundary point, so it has one more point than cells.
cell_count(loc, topo, n_points) = (loc === Face && topo !== Periodic) ? n_points - 1 : n_points

# Target CELL range needed to reconstruct the field points `first_point:last_point`.
# Center points ARE cells. A Face point sits between two cells, so we also need the
# ½-stencil neighbour one cell below (Periodic: cell `first_point - 1` may be 0, i.e. the
# wrapped last cell; Bounded: clamp to the grid and rely on `boundary_faces` at the ends).
# For a Periodic dimension the range is capped at one full period (`1:n_cells`) — a wider
# box would exceed 360° — and the wrap neighbour is handled in `reconstruct_faces`.
function covering_cells(loc, topo, first_point, last_point, n_cells)
    loc === Center && return first_point:last_point
    if topo === Periodic
        (last_point - first_point + 1) >= n_cells && return 1:n_cells
        return (first_point - 1):last_point
    end
    return max(1, first_point - 1):min(n_cells, last_point)
end

# Average a host array from Face to Center along dimension `dim` (½ of the two adjacent
# faces). Periodic wraps the top face back to the first; Bounded drops the extra point.
function average_faces_to_centers(data, dim, topo)
    n_cells = topo === Periodic ? size(data, dim) : size(data, dim) - 1
    lower = selectdim(data, dim, 1:n_cells)
    upper = topo === Periodic ? selectdim(data, dim, [mod1(i + 1, size(data, dim)) for i in 1:n_cells]) :
                                selectdim(data, dim, 2:(n_cells + 1))
    return 0.5 .* (lower .+ upper)
end

# All-`Center` host proxy of a (possibly Face-located) source field's interior data.
function center_proxy(data, loc, topo)
    loc[1] === Face && (data = average_faces_to_centers(data, 1, topo[1]))
    loc[2] === Face && (data = average_faces_to_centers(data, 2, topo[2]))
    return data
end

# Reconstruct the Face points `first_point:last_point` in dimension `dim` from a Center
# buffer whose first slab along `dim` is cell `first_cell`. Interior faces are the
# ½-average of the two neighbouring centers (the Periodic wrap is already baked into
# `first_cell`). The two domain-boundary faces of a Bounded dimension follow `boundary_faces`.
function reconstruct_faces(centers, dim, topo, first_point, last_point, first_cell, n_cells, boundary_faces)
    face_size = collect(size(centers))
    face_size[dim] = last_point - first_point + 1
    faces = similar(centers, face_size...)

    for (offset, point) in enumerate(first_point:last_point)
        face = selectdim(faces, dim, offset)
        if topo !== Periodic && point == 1
            face .= boundary_faces === :zero ? zero(eltype(centers)) : selectdim(centers, dim, 1 - first_cell + 1)
        elseif topo !== Periodic && point == n_cells + 1
            face .= boundary_faces === :zero ? zero(eltype(centers)) : selectdim(centers, dim, n_cells - first_cell + 1)
        else
            below_cell = point - 1
            # Periodic wrap: if the below neighbour isn't in the buffer (full-period case),
            # it is the last cell (cell 0 ≡ cell N). Sub-range buffers include cell 0 directly.
            (topo === Periodic && below_cell < first_cell) && (below_cell += n_cells)
            below = selectdim(centers, dim, below_cell - first_cell + 1)
            above = selectdim(centers, dim, point - first_cell + 1)
            face .= 0.5 .* (below .+ above)
        end
    end
    return faces
end

# Conservative two-pass regrid of an all-`Center` source host array onto the target CENTER
# cells `x_cells × y_cells` (index ranges; they may fall outside `[1, target_nx]` when the
# x-topology is Periodic, in which case the source is gathered with wraparound). Returns a
# host array of the regridded centers.
function regrid_centers(source_centers, bounds, Nz, halo, regrid_arch,
                        x_topo, source_nx, target_nx, x_cells, source_ny, target_ny, y_cells)
    λbounds, φbounds, zbounds = bounds
    first_i, last_i = first(x_cells), last(x_cells)
    first_j, last_j = first(y_cells), last(y_cells)
    n_target_i, n_target_j = length(x_cells), length(y_cells)

    Δλ_target = (λbounds[2] - λbounds[1]) / target_nx
    Δφ_target = (φbounds[2] - φbounds[1]) / target_ny
    Δλ_source = (λbounds[2] - λbounds[1]) / source_nx
    Δφ_source = (φbounds[2] - φbounds[1]) / source_ny

    # Source cells covering the (possibly wrapped) target box, rounded OUTWARD to whole
    # cells — `fld`/`cld` are exact and reduce to `÷` when the box lands on source faces.
    source_i0 = fld((first_i - 1) * source_nx, target_nx) + 1
    source_i1 = cld(last_i * source_nx, target_nx)
    source_j0 = fld((first_j - 1) * source_ny, target_ny) + 1
    source_j1 = cld(last_j * source_ny, target_ny)

    target_longitude = (λbounds[1] + (first_i - 1) * Δλ_target,   λbounds[1] + last_i * Δλ_target)
    target_latitude  = (φbounds[1] + (first_j - 1) * Δφ_target,   φbounds[1] + last_j * Δφ_target)
    source_longitude = (λbounds[1] + (source_i0 - 1) * Δλ_source, λbounds[1] + source_i1 * Δλ_source)
    source_latitude  = (φbounds[1] + (source_j0 - 1) * Δφ_source, φbounds[1] + source_j1 * Δφ_source)

    # x may wrap when Periodic; y is Bounded here, so its indices stay in range.
    source_i = x_topo === Periodic ? [mod1(i, source_nx) for i in source_i0:source_i1] : collect(source_i0:source_i1)
    source_patch = source_centers[source_i, collect(source_j0:source_j1), :]

    # Use the model halo for parity, but never larger than the smallest sub-grid extent in
    # each dimension (conservative regrid only needs 1; the extra width is just realism).
    sub_halo = (min(halo[1], length(source_i), n_target_i),
                min(halo[2], size(source_patch, 2), n_target_j),
                min(halo[3], Nz))

    # Force Bounded topology: the sub-grids are plain conservative-remap grids with no
    # periodic seam handling (a 360° box would otherwise auto-infer Periodic and zero-fill
    # the seam). Periodicity is handled explicitly via the wrapped source gather above and
    # the wrap in `reconstruct_faces`, so whole-field and per-shard results agree.
    make_grid(nx, ny, longitude, latitude) =
        LatitudeLongitudeGrid(regrid_arch; size = (nx, ny, Nz), halo = sub_halo,
                              longitude, latitude, z = zbounds, topology = (Bounded, Bounded, Bounded))

    coarse    = CenterField(make_grid(size(source_patch, 1), size(source_patch, 2), source_longitude, source_latitude))
    x_refined = CenterField(make_grid(n_target_i,            size(source_patch, 2), target_longitude, source_latitude))
    refined   = CenterField(make_grid(n_target_i,            n_target_j,            target_longitude, target_latitude))

    interior(coarse) .= source_patch
    regrid!(x_refined, coarse)     # x-pass: source_longitude → target_longitude  (y, z identical)
    regrid!(refined, x_refined)    # y-pass: source_latitude  → target_latitude   (x, z identical)
    return Array(interior(refined))
end

"""
    regrid_shard(source_field, target_slice, target_size; regrid_arch=CPU(), boundary_faces=:zero)

Produce the host buffer for the target shard spanning global index box `target_slice`,
upscaled from `source_field`. Handles arbitrary `(Center|Face)` locations in x and y:
Face dimensions are averaged to a `Center` proxy, conservatively regridded, then
reconstructed to faces. `boundary_faces` (`:zero` or `:extrapolate`) sets the two
domain-boundary faces of a Face-in-Bounded dimension (e.g. `v` at ±lat).

Only the source cells overlapping the box are read (rounded outward; periodic wrap in x
handled), so no global field is built. z is index pass-through (`Center` `Nz`, `Face`
`Nz+1` like `w`, or 2D `Nothing`), regridded level-by-level; source and target must share
the vertical grid.
"""
function regrid_shard(source_field, target_slice, target_size; regrid_arch = CPU(), boundary_faces = :zero)
    loc = location(source_field)
    source_grid = source_field.grid
    topo = topology(source_grid)
    bounds = latlon_bounds(source_grid)
    Nz = size(source_field, 3)
    halo = (source_grid.Hx, source_grid.Hy, source_grid.Hz)   # model halo; capped to sub-grid size in regrid_centers

    @assert Nz == target_size[3] "z is pass-through; source and target must share the vertical grid (got source Nz=$Nz, target $(target_size[3]))"

    source_nx = cell_count(loc[1], topo[1], size(source_field, 1))
    source_ny = cell_count(loc[2], topo[2], size(source_field, 2))
    target_nx = cell_count(loc[1], topo[1], target_size[1])
    target_ny = cell_count(loc[2], topo[2], target_size[2])

    # 1. All-Center proxy of the source (Face dimensions averaged down).
    source_centers = center_proxy(Array(interior(source_field)), loc, topo)

    # 2. Target cells needed to produce this shard's field points (plus face-stencil neighbours).
    first_i, last_i = first(target_slice[1]), last(target_slice[1])
    first_j, last_j = first(target_slice[2]), last(target_slice[2])
    x_cells = covering_cells(loc[1], topo[1], first_i, last_i, target_nx)
    y_cells = covering_cells(loc[2], topo[2], first_j, last_j, target_ny)

    # 3. Conservative two-pass regrid of the proxy onto those target cells.
    centers = regrid_centers(source_centers, bounds, Nz, halo, regrid_arch,
                             topo[1], source_nx, target_nx, x_cells, source_ny, target_ny, y_cells)

    # 4. Reconstruct faces on Face dimensions (Center dimensions pass straight through).
    loc[1] === Face && (centers = reconstruct_faces(centers, 1, topo[1], first_i, last_i, first(x_cells), target_nx, boundary_faces))
    loc[2] === Face && (centers = reconstruct_faces(centers, 2, topo[2], first_j, last_j, first(y_cells), target_ny, boundary_faces))
    return centers
end

"""
    assemble_sharded_field!(target_field, source_field, arch; regrid_arch=CPU(), boundary_faces=:zero)

Fill a distributed `target_field` by regridding `source_field` shard-by-shard and
assembling the shards into a sharded `ConcreteIFRTArray` — with no cross-process
all-gather and without ever building the global fine field on one device.

Reads the target field's **own** sharding off its data array (the exact inverse of
`local_shards_to_host`) and assembles the FULL haloed array against it, rather than
reconstructing a `DimsSharding`. That is essential for Face-located fields (e.g. `v`,
whose `Ny+1` interior view reshards to a layout a hand-built `DimsSharding` does not
match). Each shard's buffer is zero in the halo band and regridded in the interior
overlap; `fill_halo_regions!` then fixes the halos. Each process builds only the shards
its devices address.
"""
function assemble_sharded_field!(target_field, source_field, arch; regrid_arch = CPU(), boundary_faces = :zero)
    target_data = Reactant.ancestor(target_field.data)              # full haloed sharded array
    reactant_sharding = Sharding.unwrap_shardinfo(target_data.sharding)
    global_shape = size(target_data)

    if reactant_sharding isa Sharding.HloSharding
        (; hlo_sharding) = reactant_sharding
    else
        (; hlo_sharding) = Sharding.HloSharding(reactant_sharding, global_shape)
    end

    client = Reactant.XLA.client(Reactant.XLA.synced_buffer(target_data.data))
    all_devices = Reactant.XLA.get_device.((client,), reactant_sharding.mesh.device_ids)
    all_slices, _ = Reactant.XLA.sharding_to_concrete_array_indices(
        convert(Reactant.XLA.CondensedOpSharding, hlo_sharding),
        global_shape, reactant_sharding.mesh.logical_device_ids)
    local_slices = [slice for (device, slice) in zip(all_devices, all_slices) if Reactant.XLA.is_addressable(device)]

    interior_size = size(target_field)
    halos = ntuple(d -> (global_shape[d] - interior_size[d]) ÷ 2, 3)   # 0 for reduced (2D) dims
    FT = eltype(target_data)

    shard_buffers = map(local_slices) do shard_slice
        buffer = zeros(FT, map(length, shard_slice)...)
        overlap = ntuple(d -> intersect(shard_slice[d], (halos[d] + 1):(halos[d] + interior_size[d])), 3)
        if all(!isempty, overlap)
            point_slice   = ntuple(d -> (first(overlap[d]) - halos[d]):(last(overlap[d]) - halos[d]), 3)
            buffer_slice  = ntuple(d -> (first(overlap[d]) - first(shard_slice[d]) + 1):(last(overlap[d]) - first(shard_slice[d]) + 1), 3)
            buffer[buffer_slice...] = regrid_shard(source_field, point_slice, interior_size; regrid_arch, boundary_faces)
        end
        buffer
    end

    assembled = Reactant.ConcreteIFRTArray(shard_buffers, global_shape; client, sharding = reactant_sharding)
    target_data .= assembled
    fill_halo_regions!(target_field)
    return target_field
end

"""
    regrid_field!(target_field, source_field; regrid_arch=CPU(), boundary_faces=:zero)

Upscale one field from `source_field` onto `target_field` in place, dispatching on
`target_field`'s architecture: a sharded no-gather assembly on a `Distributed`
architecture, or a single local regrid otherwise. Halos are filled.
"""
function regrid_field!(target_field, source_field; regrid_arch = CPU(), boundary_faces = :zero)
    arch = architecture(target_field.grid)
    if arch isa Oceananigans.Distributed
        return assemble_sharded_field!(target_field, source_field, arch; regrid_arch, boundary_faces)
    end
    interior_size = size(target_field)
    whole_field = ntuple(d -> 1:interior_size[d], 3)
    interior(target_field) .= regrid_shard(source_field, whole_field, interior_size; regrid_arch, boundary_faces)
    fill_halo_regions!(target_field)
    return target_field
end

"""
    upscale_prognostic_fields!(target_model, source_model; regrid_arch=CPU(), boundary_faces=:zero) -> Vector{Symbol}

Upscale every prognostic field from `source_model` onto `target_model` in place, with no
host gather. Cell-centered tracers use the conservative path; Face-located velocity/
free-surface fields (`u`, `v`, `U`, `V`) go through the center-proxy round-trip with
face reconstruction (`boundary_faces` sets the domain-boundary faces of Bounded-Face
dimensions). Returns the names that were upscaled.

Only fields present in both models with matching staggering are touched. For a rest
initial condition the velocities are zero, so upscaling them is a no-op in value.
"""
function upscale_prognostic_fields!(target_model, source_model; regrid_arch = CPU(), boundary_faces = :zero)
    target_fields = prognostic_fields(target_model)
    source_fields = prognostic_fields(source_model)
    upscaled = Symbol[]
    for name in keys(target_fields)
        haskey(source_fields, name) || continue
        target_field, source_field = target_fields[name], source_fields[name]
        location(target_field) === location(source_field) || continue   # same staggering required
        regrid_field!(target_field, source_field; regrid_arch, boundary_faces)
        push!(upscaled, name)
    end
    return upscaled
end
