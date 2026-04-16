#!/usr/bin/env julia
#
# Assemble & plot xy-slices from per-rank sharded field data.
#
# Handles multi-rank (multi-node) output by reading all fields_rank*.dat
# files, placing each shard into the global array, and stripping halos.
#
# Usage (inside uenv):
#   julia --project=.. --startup-file=no plot_grad_xy.jl [--quantile=0.02] [PATH] [Z_LAYERS...]
#
# PATH         — directory containing fields_rank*.dat files, or auto-detected.
# Z_LAYERS     — 1-based z-levels to plot. Default: all saved levels.
# --quantile=Q — clip colorbar at Qth / (1-Q)th percentile (default: 0.02 = 2nd–98th pct)

using Serialization
using CairoMakie
using Printf
using Reactant

# ─── Safe scalar conversion ──────────────────────────────────────────

function _is_reactant_type(x)
    mod = parentmodule(typeof(x))
    mod_str = string(mod)
    return contains(mod_str, "Reactant") || contains(mod_str, "IFRT")
end

function safe_scalar(x)
    _is_reactant_type(x) && return NaN
    x isa AbstractFloat && return Float64(x)
    x isa Integer && return Int(x)
    try; return Float64(x); catch; end
    NaN
end

# ─── Auto-discover most recent output ────────────────────────────────

function find_latest_output()
    runs_dir = joinpath(@__DIR__, "runs")
    isdir(runs_dir) || return nothing

    for run in sort(readdir(runs_dir; join=true); by=mtime, rev=true)
        base = joinpath(run, "output", "differentiable_sim")
        isdir(base) || continue

        for sub in sort(readdir(base; join=true); by=mtime, rev=true)
            isdir(sub) || continue
            for leaf in ("gradients", "initial_fields")
                d = joinpath(sub, leaf)
                if isdir(d) && any(f -> startswith(f, "fields_rank"), readdir(d))
                    return d
                end
            end
            if any(f -> startswith(f, "fields_rank"), readdir(sub))
                return sub
            end
        end

        for leaf in ("gradients", "initial_fields")
            d = joinpath(base, leaf)
            if isdir(d) && any(f -> startswith(f, "fields_rank"), readdir(d))
                return d
            end
        end

        legacy = joinpath(base, "grad_output.dat")
        isfile(legacy) && return legacy
    end
    return nothing
end

# ─── Parse CLI args ──────────────────────────────────────────────────

global input_path = nothing
global z_args = Int[]
global QCLIP = 0.02

for a in ARGS
    if startswith(a, "--quantile=")
        global QCLIP = parse(Float64, split(a, '=')[2])
    elseif input_path === nothing && (isdir(a) || isfile(a))
        global input_path = a
    else
        try; push!(z_args, parse(Int, a)); catch; end
    end
end

if input_path === nothing
    global input_path = find_latest_output()
    input_path !== nothing || error(
        "No gradient output found. Pass the path explicitly:\n" *
        "  julia --project=.. plot_grad_xy.jl <dir-with-fields_rank-files> [z1 z2 ...]")
end

if isfile(input_path) && startswith(basename(input_path), "fields_rank")
    input_path = dirname(input_path)
end

# ─── Assemble global fields from per-rank shards ─────────────────────

function load_and_assemble(dir)
    rank_files = sort(filter(
        f -> startswith(basename(f), "fields_rank") && endswith(f, ".dat"),
        readdir(dir; join=true)))
    isempty(rank_files) && error("No fields_rank*.dat in $dir")

    nranks = length(rank_files)
    @info "Found $nranks rank file(s) in $dir"

    header = open(first(rank_files)) do io; Serialization.deserialize(io); end
    hs = let v = get(header, :halo_sizes, (0, 0, 0)); v isa Tuple ? v : Tuple(v); end
    slice_info = get(header, :slices, nothing)
    field_names = header[:field_names]

    meta_path = joinpath(dir, "meta.dat")
    meta = isfile(meta_path) ? open(meta_path) do io; Serialization.deserialize(io); end : Dict{Symbol,Any}()

    fields = Dict{Symbol, Array}()

    for fname in field_names
        first_info = header[fname]
        T = eltype(first(first_info.local_arrays))
        global_arr = zeros(T, first_info.global_shape...)

        for rf in rank_files
            state = open(rf) do io; Serialization.deserialize(io); end
            fi = state[fname]
            for (arr, sb) in zip(fi.local_arrays, fi.local_slices)
                global_arr[Tuple(a:b for (a, b) in sb)...] = arr
            end
        end

        sliced_dim = if slice_info !== nothing
            si = findfirst(s -> Symbol("$(s.field)_$(s.plane)") === fname, slice_info)
            si !== nothing ? slice_info[si].dim : nothing
        else
            nothing
        end

        interior_ranges = ntuple(ndims(global_arr)) do d
            h = d <= 3 ? hs[d] : 0
            (sliced_dim !== nothing && d == sliced_dim) ? (1:size(global_arr, d)) :
            h > 0 ? ((h+1):(size(global_arr, d) - h)) : (1:size(global_arr, d))
        end
        fields[fname] = global_arr[interior_ranges...]

        sz = size(fields[fname])
        @info "  $fname: global=$(first_info.global_shape) → interior=$sz  ($(nranks) rank(s))"
    end

    z_label = get(meta, :z_levels, nothing)
    return (
        fields   = fields,
        loss_val = safe_scalar(get(meta, :loss_value, NaN)),
        nsteps   = safe_scalar(get(meta, :nsteps, 0)),
        Nz       = safe_scalar(get(meta, :Nz, 0)),
        z_label  = z_label,
    )
end

# ─── Load legacy grad_output.dat format ──────────────────────────────

function load_legacy_format(path)
    state = open(path) do io; Serialization.deserialize(io); end
    grad_data = state[:gradients]
    z_saved = get(state, :z_levels, nothing)
    return (
        fields   = grad_data,
        loss_val = safe_scalar(get(state, :loss_value, NaN)),
        nsteps   = safe_scalar(get(state, :nsteps, 0)),
        Nz       = safe_scalar(get(state, :Nz, 0)),
        z_label  = z_saved,
    )
end

# ─── Load ─────────────────────────────────────────────────────────────

data = if isdir(input_path)
    @info "Loading shard format" dir=input_path
    load_and_assemble(input_path)
else
    @info "Loading legacy format" file=input_path
    load_legacy_format(input_path)
end

OUT_DIR = joinpath(dirname(input_path), "plots")
mkpath(OUT_DIR)

fields   = data.fields
loss_val = data.loss_val
nsteps   = Int(data.nsteps)
Nz       = Int(data.Nz)
z_label  = data.z_label

@info "Loaded" n_fields=length(fields) loss_val nsteps Nz z_label
@info "Fields: $(sort(collect(keys(fields))))"

# ─── Determine z-layers to plot ──────────────────────────────────────

first_arr = first(values(fields))
nz_saved = ndims(first_arr) >= 3 ? size(first_arr, 3) : 1

z_layers = if !isempty(z_args)
    for z in z_args
        1 <= z <= nz_saved || error("z=$z out of range [1, $nz_saved]")
    end
    z_args
else
    collect(1:nz_saved)
end

# ─── Colormap helper ─────────────────────────────────────────────────

function field_colormap(fname)
    s = string(fname)
    (contains(s, "ρw") || contains(s, "ρu") || contains(s, "ρv")) && return :balance
    contains(s, "ρθ") && return :magma
    contains(s, "ρq") && return :dense
    contains(s, "ρ") && return :viridis
    return :viridis
end

# ─── Percentile-clipped color range ──────────────────────────────────

function compute_colorrange(fin, cmap; quantile_clip=QCLIP)
    isempty(fin) && return (0.0, 1.0)

    sorted = sort(fin)
    n = length(sorted)
    lo = sorted[max(1, round(Int, quantile_clip * n))]
    hi = sorted[min(n, round(Int, (1 - quantile_clip) * n))]

    if lo ≈ hi
        lo, hi = extrema(fin)
    end
    if lo ≈ hi
        hi = lo + max(1.0, abs(lo) * 0.01)
    end

    if cmap === :balance
        sym = max(abs(lo), abs(hi))
        sym = sym > 0 ? sym : 1.0
        lo, hi = -sym, sym
    end
    return (lo, hi)
end

# ─── Plot ─────────────────────────────────────────────────────────────

for (name, arr) in sort(collect(fields); by=first)
    ndims(arr) < 2 && continue
    cmap = field_colormap(name)

    for zi in z_layers
        slab = ndims(arr) >= 3 ? arr[:, :, zi] : arr[:, :]
        fin  = filter(isfinite, vec(Float64.(slab)))

        @info "  $name  z=$zi  size=$(size(slab)) extrema=$(isempty(fin) ? (NaN,NaN) : extrema(fin))"

        lo, hi = compute_colorrange(fin, cmap)

        zlbl = z_label !== nothing && zi <= length(z_label) ? "$(z_label[zi])" : "$zi"
        loss_str = isnan(loss_val) ? "N/A" : @sprintf("%.4e", loss_val)

        fig = Figure(size=(1000, 700))
        ax  = Axis(fig[1, 1];
            title  = "$name   z=$zlbl   (nsteps=$nsteps, L=$loss_str)",
            xlabel = "λ  index", ylabel = "φ  index", aspect = DataAspect())
        heatmap!(ax, slab; colormap=cmap, colorrange=(lo, hi))
        Colorbar(fig[1, 2]; colormap=cmap, limits=(lo, hi), label=string(name))

        outpath = joinpath(OUT_DIR, "$(name)_z$(zlbl).png")
        save(outpath, fig)
        @info "  → $outpath"
    end
end

@info "Done. Plots in $OUT_DIR"
