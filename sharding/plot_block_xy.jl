#!/usr/bin/env julia
#
# Assemble global xy-slice plots from per-rank shards (128-GPU / 32-node run).
#
# Usage (inside uenv):
#   julia --project=.. --startup-file=no plot_block_xy.jl
#
# Outer loop: each (field, z_layer) combination.
# Inner loop: load each rank's data, extract the relevant slab, strip halos,
#             place into global array, then free rank data.
# After assembly, save one PNG per field to the ngpu directory.

using Serialization
using CairoMakie

const RUN_DIR   = joinpath(@__DIR__, "runs/2026-04-13T06-47-58.707_wxBa/output")
const JOB_ID    = "1849977"
const BLOCK     = "after_interpolation"
NGPU      = "4"
const NRANKS    = 1
const Z_LAYER   = 1   # index into the saved z-levels ([1,2,4,8,16] → picks level 2)
const OUT_DIR   = joinpath(@__DIR__, "runs/2026-04-13T06-47-58.707_wxBa/ngpu=00$(NGPU)/plots_$(BLOCK)")

const XY_FIELDS = [:ρ_xy, :ρu_xy, :ρv_xy, :ρw_xy, :ρθ_xy, :ρqᵛ_xy]

function field_colormap(fname)
    s = string(fname)
    contains(s, "ρw") && return :balance
    contains(s, "ρu") && return :balance
    contains(s, "ρv") && return :balance
    contains(s, "ρθ") && return :magma
    contains(s, "ρq") && return :dense
    contains(s, "ρ_") && return :viridis
    return :viridis
end

mkpath(OUT_DIR)

function load_rank_data(rank)
    dir = joinpath(RUN_DIR, "$(JOB_ID).$(rank)", BLOCK, "output")
    path = joinpath(dir, "fields_rank$(rank).dat")
    isfile(path) || error("Missing: $path")
    return open(path) do io
        Serialization.deserialize(io)
    end
end

function strip_halo_range(global_range, global_size, H)
    lo = first(global_range)
    hi = last(global_range)
    local_lo = max(1, H + 1 - lo + 1)
    local_hi = size_from_range(global_range) - max(0, hi - (global_size - H))
    return local_lo, local_hi
end
size_from_range(r) = last(r) - first(r) + 1

for fname in XY_FIELDS
    @info "=== Assembling $fname at z-layer $Z_LAYER ==="

    global_shape = nothing
    halo = nothing
    assembled = nothing

    for rank in 0:(NRANKS - 1)
        state = try
            load_rank_data(rank)
        catch e
            @warn "Skipping rank $rank" exception=e
            continue
        end

        hs = get(state, :halo_sizes, (0, 0, 0))
        hs = hs isa Tuple ? hs : Tuple(hs)

        haskey(state, fname) || continue
        info = state[fname]

        if global_shape === nothing
            global_shape = info.global_shape
            halo = hs
            Hx, Hy = hs[1], hs[2]
            interior_nx = global_shape[1] - 2Hx
            interior_ny = global_shape[2] - 2Hy
            assembled = fill(NaN32, interior_nx, interior_ny)
            @info "  global_shape=$global_shape halo=$hs interior=($interior_nx, $interior_ny)"
        end

        Hx, Hy = halo[1], halo[2]
        gx, gy = global_shape[1], global_shape[2]

        for (arr, slice_bounds) in zip(info.local_arrays, info.local_slices)
            xr = slice_bounds[1][1]:slice_bounds[1][2]
            yr = slice_bounds[2][1]:slice_bounds[2][2]
            slab = arr[:, :, Z_LAYER]

            x_lo, x_hi = strip_halo_range(xr, gx, Hx)
            y_lo, y_hi = strip_halo_range(yr, gy, Hy)

            trimmed = slab[x_lo:x_hi, y_lo:y_hi]

            dst_x_start = first(xr) + x_lo - 1 - Hx
            dst_y_start = first(yr) + y_lo - 1 - Hy
            dst_xr = dst_x_start:(dst_x_start + size(trimmed, 1) - 1)
            dst_yr = dst_y_start:(dst_y_start + size(trimmed, 2) - 1)

            assembled[dst_xr, dst_yr] .= trimmed
        end

        state = nothing
        GC.gc(false)
    end

    if assembled === nothing
        @warn "No data found for $fname"
        continue
    end

    nnan = count(isnan, assembled)
    fin = filter(isfinite, vec(assembled))
    @info "  Assembled: size=$(size(assembled)) NaN=$nnan/$(length(assembled)) extrema=$(isempty(fin) ? (NaN,NaN) : extrema(fin))"

    fig = Figure(size=(1200, 800))
    ax = Axis(fig[1, 1]; title="$fname  z-layer=$Z_LAYER  ($BLOCK, $NGPU GPUs)",
              xlabel="longitude index", ylabel="latitude index", aspect=DataAspect())

    lo, hi = if isempty(fin)
        0f0, 1f0
    else
        extrema(fin)
    end
    if lo == hi; hi = lo + 1f0; end

    cmap = field_colormap(fname)
    heatmap!(ax, assembled; colormap=cmap, colorrange=(lo, hi))
    Colorbar(fig[1, 2]; colormap=cmap, limits=(lo, hi), label=string(fname))

    outpath = joinpath(OUT_DIR, "$(fname)_z$(Z_LAYER).png")
    save(outpath, fig)
    @info "  Saved $outpath"

    assembled = nothing
    GC.gc()
end

@info "Done. All plots in $OUT_DIR"
