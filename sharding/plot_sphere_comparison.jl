#!/usr/bin/env julia
#
# Side-by-side sphere plots of primal (evolved) fields and tangent (gradient)
# fields from a differentiable simulation run.
#
# Usage:
#   julia --project=.. plot_sphere_comparison.jl <ngpu-dir> [options]
#
# Options (order-independent):
#   --grad=<subdir>      gradient subdirectory (default: gradients_varw)
#   --fields=ρ,ρθ,ρqᵛ   comma-separated field base names to plot (default: all)
#   --z=1,2,3            comma-separated z-layer indices (default: all saved)
#   --dpi=300            output resolution scale (default: 300)
#   --maxpix=1500        max pixels along longest slab axis for sphere render
#   --quantile=0.02      clip colorbar at this quantile (default: 0.02 = 2nd–98th pct)
#   --smooth=1.5         Gaussian blur σ in pixels (applied after downsample; default: 1.5)
#   --azimuth=210        camera azimuth in degrees (default: 210)
#   --elevation=25       camera elevation in degrees (default: 25)
#
# Examples:
#   julia --project=.. plot_sphere_comparison.jl \
#       runs/.../output/differentiable_sim/ngpu=00128
#
#   julia --project=.. plot_sphere_comparison.jl \
#       runs/.../output/differentiable_sim/ngpu=00128 \
#       --grad=gradients_mean_theta2 --fields=ρθ,ρqᵛ --z=1,3

using Serialization
using CairoMakie
using Printf
using Statistics: mean, std

try
    @eval using Reactant
catch
end

# ═══════════════════════════════════════════════════════════════════════
# CLI parsing
# ═══════════════════════════════════════════════════════════════════════

function parse_args(args)
    opts = Dict{String,String}()
    positional = String[]
    for a in args
        if startswith(a, "--")
            kv = split(a[3:end], '='; limit=2)
            opts[kv[1]] = length(kv) == 2 ? kv[2] : "true"
        else
            push!(positional, a)
        end
    end
    return positional, opts
end

positional, opts = parse_args(ARGS)

ngpu_dir     = length(positional) >= 1 ? positional[1] : nothing
grad_subdir  = get(opts, "grad", "gradients_varw")
field_filter = let s = get(opts, "fields", ""); isempty(s) ? nothing : Symbol.(split(s, ',')); end
z_filter     = let s = get(opts, "z", ""); isempty(s) ? nothing : parse.(Int, split(s, ',')); end
DPI          = parse(Int, get(opts, "dpi", "300"))
MAX_PIX      = parse(Int, get(opts, "maxpix", "1500"))
QCLIP        = parse(Float64, get(opts, "quantile", "0.02"))
SMOOTH_SIGMA = parse(Float64, get(opts, "smooth", "1.5"))
CAM_AZ       = parse(Float64, get(opts, "azimuth", "210"))
CAM_EL       = parse(Float64, get(opts, "elevation", "25"))

# Auto-discover most recent run
if ngpu_dir === nothing
    runs_dir = joinpath(@__DIR__, "runs")
    if isdir(runs_dir)
        for run in sort(readdir(runs_dir; join=true); by=mtime, rev=true)
            base = joinpath(run, "output", "differentiable_sim")
            isdir(base) || continue
            for sub in sort(readdir(base; join=true); by=mtime, rev=true)
                isdir(sub) || continue
                ef = joinpath(sub, "evolved_fields", "output")
                gf = joinpath(sub, grad_subdir, "output")
                if isdir(ef) && isdir(gf)
                    ngpu_dir = sub
                    break
                end
            end
            ngpu_dir !== nothing && break
        end
    end
end
ngpu_dir !== nothing || error(
    "No run directory found. Pass the ngpu directory explicitly:\n" *
    "  julia --project=.. plot_sphere_comparison.jl <ngpu-dir>")

primal_dir  = joinpath(ngpu_dir, "evolved_fields", "output")
tangent_dir = joinpath(ngpu_dir, grad_subdir, "output")
isdir(primal_dir)  || error("Primal dir not found: $primal_dir")
isdir(tangent_dir) || error("Tangent dir not found: $tangent_dir")

OUT_DIR = joinpath(ngpu_dir, "sphere_plots_$(grad_subdir)")
mkpath(OUT_DIR)

@info "Primal  : $primal_dir"
@info "Tangent : $tangent_dir"
@info "Output  : $OUT_DIR"

# ═══════════════════════════════════════════════════════════════════════
# Data loading (reused from plot_grad_xy.jl)
# ═══════════════════════════════════════════════════════════════════════

function safe_scalar(x)
    x isa AbstractFloat && return Float64(x)
    x isa Integer && return Int(x)
    try; return Float64(x); catch; end
    NaN
end

function load_and_assemble(dir)
    rank_files = sort(filter(
        f -> startswith(basename(f), "fields_rank") && endswith(f, ".dat"),
        readdir(dir; join=true)))
    isempty(rank_files) && error("No fields_rank*.dat in $dir")

    nranks = length(rank_files)
    @info "  Found $nranks rank file(s) in $dir"

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
    end

    z_label = get(meta, :z_levels, nothing)
    return (fields=fields, z_label=z_label,
            loss_val=safe_scalar(get(meta, :loss_value, NaN)),
            nsteps=safe_scalar(get(meta, :nsteps, 0)))
end

@info "Loading primal fields..."
primal_data  = load_and_assemble(primal_dir)
@info "Loading tangent fields..."
tangent_data = load_and_assemble(tangent_dir)

primal_fields  = primal_data.fields
tangent_fields = tangent_data.fields
z_label        = primal_data.z_label

@info "Primal fields:  $(sort(collect(keys(primal_fields))))"
@info "Tangent fields: $(sort(collect(keys(tangent_fields))))"

# ═══════════════════════════════════════════════════════════════════════
# Match fields, determine z-layers
# ═══════════════════════════════════════════════════════════════════════

common_keys = sort(collect(intersect(keys(primal_fields), keys(tangent_fields))))
isempty(common_keys) && error("No matching field names between primal and tangent")

if field_filter !== nothing
    filtered = Symbol[]
    for fk in common_keys
        base = Symbol(replace(string(fk), r"_xy$" => ""))
        base in field_filter && push!(filtered, fk)
    end
    common_keys = filtered
    isempty(common_keys) && error("No fields match filter: $field_filter")
end

first_arr = primal_fields[first(common_keys)]
nz_saved = ndims(first_arr) >= 3 ? size(first_arr, 3) : 1

z_layers = if z_filter !== nothing
    for z in z_filter; 1 <= z <= nz_saved || error("z=$z out of range [1,$nz_saved]"); end
    z_filter
else
    collect(1:nz_saved)
end

@info "Plotting $(length(common_keys)) field(s) × $(length(z_layers)) z-layer(s)"

# ═══════════════════════════════════════════════════════════════════════
# Plotting utilities
# ═══════════════════════════════════════════════════════════════════════

const FIELD_LABELS = Dict(
    :ρ => "ρ", :ρu => "ρu", :ρv => "ρv", :ρw => "ρw", :ρθ => "ρθ",
    :ρqᵛ => "ρqᵛ", :ρqᶜˡ => "ρqᶜˡ", :ρqᶜⁱ => "ρqᶜⁱ", :ρqʳ => "ρqʳ", :ρqˢ => "ρqˢ",
)

function pretty_label(fname::Symbol; derivative=false)
    base = Symbol(replace(string(fname), r"_xy$" => ""))
    lbl = get(FIELD_LABELS, base, string(base))
    derivative ? "∂J / ∂($lbl)" : lbl
end

function field_colormap(fname)
    s = string(fname)
    (contains(s, "ρw") || contains(s, "ρu") || contains(s, "ρv")) && return :balance
    contains(s, "ρθ") && return :magma
    contains(s, "ρq") && return :dense
    contains(s, "ρ")  && return :viridis
    return :viridis
end

function gaussian_smooth(slab::Matrix{Float64}, σ::Float64)
    σ <= 0 && return slab
    r = ceil(Int, 3σ)
    kern = [exp(-(i^2 + j^2) / (2σ^2)) for i in -r:r, j in -r:r]
    kern ./= sum(kern)
    nx, ny = size(slab)
    kx, ky = size(kern)
    hx, hy = kx ÷ 2, ky ÷ 2
    out = copy(slab)
    for j in 1:ny, i in 1:nx
        s = 0.0; w = 0.0
        for dj in -hy:hy, di in -hx:hx
            ii = mod1(i + di, nx)
            jj = clamp(j + dj, 1, ny)
            v = slab[ii, jj]
            isfinite(v) || continue
            k = kern[di + hx + 1, dj + hy + 1]
            s += k * v; w += k
        end
        out[i, j] = w > 0 ? s / w : slab[i, j]
    end
    return out
end

function downsample(slab, maxpix)
    nx, ny = size(slab)
    ratio = max(nx, ny) / maxpix
    ratio <= 1 && return Float64.(slab)
    sx = max(1, round(Int, nx / ratio))
    sy = max(1, round(Int, ny / ratio))
    xidx = round.(Int, range(1, nx, length=sx))
    yidx = round.(Int, range(1, ny, length=sy))
    return Float64.(slab[xidx, yidx])
end

function compute_colorrange(slab, cmap; quantile_clip=0.02)
    fin = filter(isfinite, vec(Float64.(slab)))
    isempty(fin) && return (0.0, 1.0)

    sorted = sort(fin)
    n = length(sorted)
    lo = sorted[max(1, round(Int, quantile_clip * n))]
    hi = sorted[min(n, round(Int, (1 - quantile_clip) * n))]

    lo ≈ hi && (lo, hi = extrema(fin))
    lo ≈ hi && (hi = lo + max(1.0, abs(lo) * 0.01))

    if cmap === :balance
        sym = max(abs(lo), abs(hi))
        sym = sym > 0 ? sym : 1.0
        lo, hi = -sym, sym
    end
    return (lo, hi)
end

function sphere_coords(nx, ny; lat_range=(-80, 80))
    λ = range(0, 2π, length=nx)
    φ = range(deg2rad(lat_range[1]), deg2rad(lat_range[2]), length=ny)
    x = [cos(φj) * cos(λi) for λi in λ, φj in φ]
    y = [cos(φj) * sin(λi) for λi in λ, φj in φ]
    z = [sin(φj) for λi in λ, φj in φ]
    return x, y, z
end

function render_sphere!(fig, row, col, slab, cmap, crange;
                        azimuth=CAM_AZ, elevation=CAM_EL, maxpix=MAX_PIX,
                        smooth_σ=SMOOTH_SIGMA)
    ds = downsample(slab, maxpix)

    nan_mask = .!isfinite.(ds)
    if any(nan_mask)
        fin = filter(isfinite, vec(ds))
        fill_val = isempty(fin) ? 0.0 : mean(fin)
        ds[nan_mask] .= fill_val
    end

    if smooth_σ > 0
        ds = gaussian_smooth(ds, smooth_σ)
    end

    nx, ny = size(ds)
    sx, sy, sz = sphere_coords(nx, ny)

    ax = LScene(fig[row, col]; show_axis=false)
    surface!(ax, sx, sy, sz;
        color=ds, colormap=cmap, colorrange=crange,
        shading=NoShading, backlight=0f0)

    az_rad = deg2rad(azimuth)
    el_rad = deg2rad(elevation)
    r = 2.4
    eye = Point3f(
        r * cos(el_rad) * cos(az_rad),
        r * cos(el_rad) * sin(az_rad),
        r * sin(el_rad))
    cam = cameracontrols(ax.scene)
    cam.lookat[]      = Point3f(0, 0, 0)
    cam.eyeposition[] = eye
    cam.upvector[]    = Point3f(0, 0, 1)
    update_cam!(ax.scene, cam)

    return ax
end

# ═══════════════════════════════════════════════════════════════════════
# Main plot loop
# ═══════════════════════════════════════════════════════════════════════

loss_str = let v = tangent_data.loss_val
    isnan(v) ? "N/A" : @sprintf("%.4e", v)
end

for fname in common_keys
    p_arr = primal_fields[fname]
    t_arr = tangent_fields[fname]
    cmap  = field_colormap(fname)

    for zi in z_layers
        p_slab = ndims(p_arr) >= 3 ? p_arr[:, :, zi] : p_arr[:, :]
        t_slab = ndims(t_arr) >= 3 ? t_arr[:, :, zi] : t_arr[:, :]

        p_range = compute_colorrange(p_slab, cmap; quantile_clip=QCLIP)
        t_range = compute_colorrange(t_slab, cmap; quantile_clip=QCLIP)

        zlbl = z_label !== nothing && zi <= length(z_label) ? "$(z_label[zi])" : "$zi"
        p_label = pretty_label(fname; derivative=false)
        t_label = pretty_label(fname; derivative=true)

        p_fin = filter(isfinite, vec(Float64.(p_slab)))
        t_fin = filter(isfinite, vec(Float64.(t_slab)))
        @info "  $fname z=$zlbl" primal_range=(isempty(p_fin) ? "NaN" : @sprintf("[%.3e,%.3e]", extrema(p_fin)...)) tangent_range=(isempty(t_fin) ? "NaN" : @sprintf("[%.3e,%.3e]", extrema(t_fin)...))

        fig = Figure(size=(2200, 1000), fontsize=20,
                     figure_padding=(20, 20, 10, 10))

        # ─── Sphere panels ────────────────────────────────────
        render_sphere!(fig, 1, 1, p_slab, cmap, p_range)
        render_sphere!(fig, 1, 3, t_slab, cmap, t_range)

        # ─── Colorbars ────────────────────────────────────────
        Colorbar(fig[1, 2]; colormap=cmap, limits=p_range,
            width=18, ticklabelsize=14, flipaxis=false)
        Colorbar(fig[1, 4]; colormap=cmap, limits=t_range,
            width=18, ticklabelsize=14, flipaxis=false)

        # ─── Panel labels ─────────────────────────────────────
        Label(fig[2, 1], p_label, fontsize=24, font=:bold, halign=:center)
        Label(fig[2, 3], t_label, fontsize=24, font=:bold, halign=:center)

        # ─── Column sizing ────────────────────────────────────
        colsize!(fig.layout, 1, Relative(0.42))
        colsize!(fig.layout, 2, Relative(0.06))
        colsize!(fig.layout, 3, Relative(0.42))
        colsize!(fig.layout, 4, Relative(0.06))
        rowsize!(fig.layout, 1, Relative(0.92))

        fname_clean = replace(string(fname), "." => "_")
        outpath = joinpath(OUT_DIR, "$(fname_clean)_z$(zlbl).png")
        save(outpath, fig; px_per_unit=DPI / 72)
        @info "  → $outpath"
    end
end

@info "Done. Sphere comparison plots in $OUT_DIR"
