using Reactant
using Reactant: Sharding
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Fields: location, interior, regrid!, Field
using Oceananigans.Grids: Center, Face, x_domain, y_domain, z_domain
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
##### exact. Each shard builds a Bounded sub-grid over its own lon/lat box — which also
##### sidesteps the periodic-seam zero-fill a single global (0,360) regrid would suffer.
#####
##### Scope: correct for cell-centered `(Center, Center, Center)` fields (tracers). The
##### source and target must share the same physical domain; any refinement ratio works
##### (the source cell range covering each shard box is rounded outward, so `regrid!`
##### blends partial overlaps at the edges). Staggered velocity fields (Face locations,
##### shared boundary faces, Ny+1 counts in a Bounded dimension) and 2D free-surface
##### fields are NOT handled here — see `upscale_prognostic_fields!`.

"Physical extent of a LatitudeLongitudeGrid as `((λ₁,λ₂), (φ₁,φ₂), (z₁,z₂))`."
latlon_bounds(grid) = (x_domain(grid), y_domain(grid), z_domain(grid))

"""
    regrid_shard(source_field, target_slice, target_size; regrid_arch=CPU())

Regrid the subregion of `source_field` covering the global target index box
`target_slice` (a tuple of index ranges, one per dimension) onto that box at the
*target* resolution implied by `target_size`. Returns a plain host `Array` holding the
shard's interior.

Only the source cells overlapping the box are read, so no global field is built. Any
refinement is allowed: the source cell range covering the box is rounded OUTWARD (`fld`/
`cld`, exact integer arithmetic), so the two-pass conservative `regrid!` blends partial
overlaps at the box edges — the correct conservative answer, with no alignment required.
"""
function regrid_shard(source_field, target_slice, target_size; regrid_arch = CPU())
    loc = location(source_field)
    src_grid = source_field.grid
    (λb, φb, zb) = latlon_bounds(src_grid)
    Nz = size(source_field, 3)
    halo = (src_grid.Hx, src_grid.Hy, src_grid.Hz)

    Nxg, Nyg = target_size[1], target_size[2]
    nsx, nsy = size(source_field, 1), size(source_field, 2)

    i0, i1 = first(target_slice[1]), last(target_slice[1])
    j0, j1 = first(target_slice[2]), last(target_slice[2])
    ntx, nty = length(target_slice[1]), length(target_slice[2])

    # Source cells covering this shard's target box, rounded OUTWARD to whole cells.
    # `fld`/`cld` are exact and reduce to the aligned `÷` when the box lands on faces.
    si0 = fld((i0 - 1) * nsx, Nxg) + 1;  si1 = cld(i1 * nsx, Nxg)
    sj0 = fld((j0 - 1) * nsy, Nyg) + 1;  sj1 = cld(j1 * nsy, Nyg)

    Δλt, Δφt = (λb[2] - λb[1]) / Nxg, (φb[2] - φb[1]) / Nyg   # target spacing
    Δλs, Δφs = (λb[2] - λb[1]) / nsx, (φb[2] - φb[1]) / nsy   # source spacing
    # Exact target box, and the (⊇) whole-source-cell box that covers it.
    Xt = (λb[1] + (i0 - 1) * Δλt, λb[1] + i1 * Δλt)
    Yt = (φb[1] + (j0 - 1) * Δφt, φb[1] + j1 * Δφt)
    Xs = (λb[1] + (si0 - 1) * Δλs, λb[1] + si1 * Δλs)
    Ys = (φb[1] + (sj0 - 1) * Δφs, φb[1] + sj1 * Δφs)

    # z is index pass-through in the x/y regrid, so a uniform z on the sub-grids is fine.
    # `sub` lives on the source-cell box (Xs, Ys); the refined dims move to the target box
    # (Xt) while the not-yet-refined dim keeps the source box, so each `regrid!` pass has
    # bit-identical faces in its non-regridded dimensions.
    grid(nx, ny, lon, lat) = LatitudeLongitudeGrid(regrid_arch; size = (nx, ny, Nz), halo,
                                                   longitude = lon, latitude = lat, z = zb)
    sub = Field(loc, grid(si1 - si0 + 1, sj1 - sj0 + 1, Xs, Ys))
    tmp = Field(loc, grid(ntx,           sj1 - sj0 + 1, Xt, Ys))   # x refined, y at source res
    tgt = Field(loc, grid(ntx,           nty,           Xt, Yt))   # x and y refined
    interior(sub) .= view(interior(source_field), si0:si1, sj0:sj1, :)
    regrid!(tmp, sub)   # x-pass: Xs → Xt   (y,z identical: source-res over Ys)
    regrid!(tgt, tmp)   # y-pass: Ys → Yt   (x,z identical: target-res over Xt)
    return Array(interior(tgt))
end

"""
    assemble_sharded_field!(target_field, source_field, arch; regrid_arch=CPU())

Fill a distributed `target_field` by regridding `source_field` shard-by-shard and
assembling the shards into a sharded `ConcreteIFRTArray` — with no cross-process
all-gather and without ever building the global fine field on one device.

`arch` must be an `Oceananigans.Distributed` architecture; each process builds only the
shards its devices address (via `is_addressable`). This mirrors the slice/addressable
bookkeeping in `local_shards_to_host`.
"""
function assemble_sharded_field!(target_field, source_field, arch; regrid_arch = CPU())
    client = Reactant.XLA.default_backend()
    connectivity = arch.connectivity
    sharding = Sharding.DimsSharding(connectivity, (1, 2), (:x, :y))
    gsize = size(target_field)

    (; hlo_sharding) = Sharding.HloSharding(sharding, gsize)
    all_devices = Reactant.XLA.get_device.((client,), connectivity.device_ids)
    addressable = [i - 1 for (i, d) in enumerate(all_devices) if Reactant.XLA.is_addressable(d)]
    slices, _ = Reactant.XLA.sharding_to_concrete_array_indices(hlo_sharding, gsize, addressable)

    buffers = [regrid_shard(source_field, slice, gsize; regrid_arch) for slice in slices]
    ic = Reactant.ConcreteIFRTArray(buffers, gsize; client, sharding)
    interior(target_field) .= ic
    fill_halo_regions!(target_field)
    return target_field
end

"""
    regrid_field!(target_field, source_field; regrid_arch=CPU())

Upscale one cell-centered field from `source_field` onto `target_field` in place,
dispatching on `target_field`'s architecture: a sharded no-gather assembly on a
`Distributed` architecture, or a single local regrid otherwise. Halos are filled.
"""
function regrid_field!(target_field, source_field; regrid_arch = CPU())
    arch = architecture(target_field.grid)
    if arch isa Oceananigans.Distributed
        return assemble_sharded_field!(target_field, source_field, arch; regrid_arch)
    end
    gsize = size(target_field)
    buffer = regrid_shard(source_field, map(n -> 1:n, gsize), gsize; regrid_arch)
    interior(target_field) .= buffer
    fill_halo_regions!(target_field)
    return target_field
end

"""
    upscale_prognostic_fields!(target_model, source_model; regrid_arch=CPU()) -> Vector{Symbol}

Upscale every cell-centered `(Center, Center, Center)` prognostic field (the tracers,
e.g. `T`, `S`) from `source_model` onto `target_model` in place, with no host gather.
Returns the names that were upscaled.

Staggered velocity fields (`u`, `v`) and 2D free-surface fields (`η`, `U`, `V`) are
deliberately left untouched: the verified conservative path here is exact only for
cell-centered fields, and for a rest initial condition (velocities zero) leaving them at
the model default is correct. Extending to staggered/2D fields requires handling Face
point counts and shared shard-boundary faces — the extension point noted in the module.
"""
function upscale_prognostic_fields!(target_model, source_model; regrid_arch = CPU())
    tgt = prognostic_fields(target_model)
    src = prognostic_fields(source_model)
    upscaled = Symbol[]
    for name in keys(tgt)
        haskey(src, name) || continue
        tf, sf = tgt[name], src[name]
        location(tf) == (Center, Center, Center) || continue
        regrid_field!(tf, sf; regrid_arch)
        push!(upscaled, name)
    end
    return upscaled
end
