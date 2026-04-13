#!/usr/bin/env julia
#
# Boundary diagnostic: assemble global xy-slice fields from 128-GPU sharded run,
# then print per-row statistics for the first/last N latitude rows to diagnose
# artifacts near the Bounded y-boundaries.
#
# Usage (inside uenv):
#   julia --project=.. --startup-file=no boundary_diagnostic.jl

using Serialization
using Printf
using Statistics

const RUN_DIR   = joinpath(@__DIR__, "runs/2026-04-13T06-47-58.707_wxBa/output")
const JOB_ID    = "1843204"
const BLOCK     = "block_0009"
const NRANKS    = 32
const DIAG_FIELDS = [:u_xy, :v_xy, :w_xy, :θ_xy, :qᵛ_xy]
const BOUNDARY_ROWS = 8  # examine this many rows from each edge

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
    local_hi = (hi - lo + 1) - max(0, hi - (global_size - H))
    return local_lo, local_hi
end

# --- Probe rank 0 to discover data structure ---
@info "Loading rank 0 to probe structure..."
state0 = load_rank_data(0)

hs = get(state0, :halo_sizes, (0, 0, 0))
hs = hs isa Tuple ? hs : Tuple(hs)
@info "halo_sizes = $hs"
@info "iteration  = $(get(state0, :iteration, nothing))"
@info "time       = $(get(state0, :time, nothing))"
@info "field_names = $(get(state0, :field_names, nothing))"

for fname in DIAG_FIELDS
    if haskey(state0, fname)
        info = state0[fname]
        @info "$fname: global_shape=$(info.global_shape)  n_local_arrays=$(length(info.local_arrays))  local_array_sizes=$(size.(info.local_arrays))"
    else
        @info "$fname: NOT FOUND in state"
    end
end

# Determine the z-index to use (pick the 2nd entry = z-level 2 from the saved levels)
z_idx = 2
if haskey(state0, first(DIAG_FIELDS))
    nz_saved = state0[first(DIAG_FIELDS)].global_shape[3]
    @info "Saved z-dimension has $nz_saved levels, using z_idx=$z_idx"
end

state0 = nothing
GC.gc()

# --- Assemble global fields ---
for fname in DIAG_FIELDS
    @info "\n=== Assembling $fname (z_idx=$z_idx) ==="

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

        hs_r = get(state, :halo_sizes, (0, 0, 0))
        hs_r = hs_r isa Tuple ? hs_r : Tuple(hs_r)

        haskey(state, fname) || continue
        info = state[fname]

        if global_shape === nothing
            global_shape = info.global_shape
            halo = hs_r
            Hx, Hy = hs_r[1], hs_r[2]
            interior_nx = global_shape[1] - 2Hx
            interior_ny = global_shape[2] - 2Hy
            assembled = fill(NaN32, interior_nx, interior_ny)
            @info "  global_shape=$global_shape halo=$hs_r interior=($interior_nx, $interior_ny)"
        end

        Hx, Hy = halo[1], halo[2]
        gx, gy = global_shape[1], global_shape[2]

        for (arr, slice_bounds) in zip(info.local_arrays, info.local_slices)
            xr = slice_bounds[1][1]:slice_bounds[1][2]
            yr = slice_bounds[2][1]:slice_bounds[2][2]
            slab = arr[:, :, z_idx]

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

    Nx, Ny = size(assembled)
    nnan = count(isnan, assembled)
    fin = filter(isfinite, vec(assembled))
    @info "  Assembled: size=($Nx, $Ny) NaN=$nnan/$(length(assembled)) extrema=$(isempty(fin) ? (NaN,NaN) : extrema(fin))"

    # Per-row boundary diagnostic
    println("\n--- $fname boundary rows (z_idx=$z_idx) ---")
    println("  First $BOUNDARY_ROWS rows (south pole / low latitude):")
    for j in 1:min(BOUNDARY_ROWS, Ny)
        row = filter(isfinite, vec(assembled[:, j]))
        if isempty(row)
            @printf("  y=%4d  [all NaN]\n", j)
        else
            @printf("  y=%4d  min=%+12.5e  max=%+12.5e  mean=%+12.5e  std=%10.5e  n=%d\n",
                    j, minimum(row), maximum(row), mean(row), std(row), length(row))
        end
    end

    mid = Ny ÷ 2
    println("  Middle rows (mid-latitude):")
    for j in (mid-1):(mid+2)
        row = filter(isfinite, vec(assembled[:, j]))
        if isempty(row)
            @printf("  y=%4d  [all NaN]\n", j)
        else
            @printf("  y=%4d  min=%+12.5e  max=%+12.5e  mean=%+12.5e  std=%10.5e  n=%d\n",
                    j, minimum(row), maximum(row), mean(row), std(row), length(row))
        end
    end

    println("  Last $BOUNDARY_ROWS rows (north pole / high latitude):")
    for j in max(1, Ny - BOUNDARY_ROWS + 1):Ny
        row = filter(isfinite, vec(assembled[:, j]))
        if isempty(row)
            @printf("  y=%4d  [all NaN]\n", j)
        else
            @printf("  y=%4d  min=%+12.5e  max=%+12.5e  mean=%+12.5e  std=%10.5e  n=%d\n",
                    j, minimum(row), maximum(row), mean(row), std(row), length(row))
        end
    end

    # Row-to-row diff diagnostic: detect sharp jumps
    println("\n  Row-to-row mean difference (first $(BOUNDARY_ROWS+2) rows):")
    prev_mean = NaN
    for j in 1:min(BOUNDARY_ROWS + 2, Ny)
        row = filter(isfinite, vec(assembled[:, j]))
        curr_mean = isempty(row) ? NaN : mean(row)
        diff = curr_mean - prev_mean
        if j > 1
            @printf("  Δmean(y=%d→%d) = %+12.5e\n", j-1, j, diff)
        end
        prev_mean = curr_mean
    end

    println("  Row-to-row mean difference (last $(BOUNDARY_ROWS+2) rows):")
    prev_mean = NaN
    for j in max(1, Ny - BOUNDARY_ROWS - 1):Ny
        row = filter(isfinite, vec(assembled[:, j]))
        curr_mean = isempty(row) ? NaN : mean(row)
        diff = curr_mean - prev_mean
        if j > max(1, Ny - BOUNDARY_ROWS - 1)
            @printf("  Δmean(y=%d→%d) = %+12.5e\n", j-1, j, diff)
        end
        prev_mean = curr_mean
    end

    assembled = nothing
    GC.gc()
end

@info "\nDiagnostic complete."
