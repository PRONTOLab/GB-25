#!/usr/bin/env julia
#=
Generate publication-quality figures for the Gordon Bell 2026 paper:

  Figure 1: Single-GPU roofline (ncu) — per-kernel scatter with BW/compute ceilings
  Figure 2: HBM BW utilization vs GPU count (XLA profiler) — complements weak scaling

Usage:
    julia --project=. plot_paper_figures.jl <ncu_csv>
    julia --project=. plot_paper_figures.jl <ncu_csv> --output-dir figs/

Requires CairoMakie (in profile/ project deps).
=#

using Printf
using CairoMakie

include(joinpath(@__DIR__, "analyze_ncu_flops.jl"))
include(joinpath(@__DIR__, "plot_roofline.jl"))

# ── GH200 120GB peak specs ───────────────────────────────────────────
const PEAK_BW_GBS     = 3_750.0    # practical HBM3 peak (GB/s)
const PEAK_BW_SPEC    = 4_000.0    # spec HBM3 peak (GB/s)

# ── XLA profiler data (5CCn run, loop2 phase, rank 0) ────────────────
# From: analyze_xla_roofline.jl ../sharding/runs/2026-04-09T00-19-49.653_5CCn
const XLA_DATA = (
    ngpus        = [    4,     8,    16,    64,   128,   512,  1024],
    nnodes       = [    1,     2,     4,    16,    32,   128,   256],
    xla_bw_gbs   = [1845.8, 1753.5, 1527.6, 903.2, 784.6, 867.7, 957.7],
    bw_util_pct  = [ 49.2,   46.7,   40.6,  24.0,  20.9,  23.1,  25.5],
    wall_per_ts  = [137.36, 150.38, 173.21, 306.06, 359.33, 291.46, 292.14],
    ncu_node_gfs = [4611.0, 4225.0, 3698.0, 2104.0, 1789.0, 2210.0, 2214.0],
    ncu_fp64_pct = [  3.4,    3.2,    2.8,    1.6,    1.3,    1.6,    1.7],
    ncu_agg_pflops = [0.0046, 0.0084, 0.0148, 0.0337, 0.0572, 0.2829, 0.5667],
)

# ── Figure 1: Single-GPU Roofline ────────────────────────────────────

function make_paper_roofline(agg, output_path;
                              peak_fp64=PEAK_FP64_GFLOPS, peak_bw=PEAK_BW_SPEC)
    names_vec = String[]
    ai_vec    = Float64[]
    gflops_vec = Float64[]
    flops_vec = Float64[]
    dur_vec   = Float64[]

    for (name, a) in agg
        a.ai > 0 && a.gflops > 0 || continue
        push!(names_vec, name)
        push!(ai_vec, a.ai)
        push!(gflops_vec, a.gflops)
        push!(flops_vec, a.total_flops)
        push!(dur_vec, a.duration_ms)
    end
    isempty(ai_vec) && error("No valid roofline data")

    # Aggregate program-level point
    total_flops = sum(flops_vec)
    total_dur   = sum(dur_vec)
    total_bytes = sum(a.dram_bytes for (_, a) in agg)
    prog_ai     = total_flops / total_bytes
    prog_gflops = (total_flops / 1e9) / (total_dur / 1e3)
    prog_bw     = (total_bytes / 1e9) / (total_dur / 1e3)

    ridge_ai = peak_fp64 / peak_bw

    ai_lo = 10.0^(floor(log10(minimum(ai_vec))) - 0.3)
    ai_hi = 10.0^(ceil(log10(max(maximum(ai_vec), ridge_ai))) + 0.5)
    gf_lo = 10.0^(floor(log10(minimum(gflops_vec))) - 0.3)

    # BW-limited and compute-limited ceiling segments
    bw_ai = [ai_lo, ridge_ai]
    bw_gf = peak_bw .* bw_ai
    cp_ai = [ridge_ai, ai_hi]
    cp_gf = [peak_fp64, peak_fp64]

    # Marker sizes ∝ log(FLOP)
    log_flops = log10.(max.(flops_vec, 1.0))
    min_lf, max_lf = extrema(log_flops)
    marker_sizes = max_lf > min_lf ?
        6.0 .+ 14.0 .* (log_flops .- min_lf) ./ (max_lf - min_lf) :
        fill(10.0, length(log_flops))

    # Color by duration fraction
    dur_frac = dur_vec ./ total_dur

    fig = Figure(; size=(560, 400), fontsize=11,
                  figure_padding=(4, 14, 4, 4))
    ax = Axis(fig[1, 1];
              xlabel="Arithmetic Intensity (FLOP/byte)",
              ylabel="Performance (GFLOP/s)",
              xscale=log10, yscale=log10,
              xminorticksvisible=true, yminorticksvisible=true,
              xminorticks=IntervalsBetween(9),
              yminorticks=IntervalsBetween(9),
              xlabelsize=12, ylabelsize=12)

    # Roofline ceiling
    lines!(ax, bw_ai, bw_gf; color=:firebrick, linewidth=2.0,
           label=@sprintf("HBM3: %.1f TB/s", peak_bw / 1000))
    lines!(ax, cp_ai, cp_gf; color=:steelblue, linewidth=2.0,
           label=@sprintf("FP64: %.1f TFLOP/s", peak_fp64 / 1000))

    # Ridge line
    vlines!(ax, [ridge_ai]; color=(:gray60, 0.5), linestyle=:dash, linewidth=0.8)

    # Per-kernel scatter
    sc = scatter!(ax, ai_vec, gflops_vec;
                  markersize=marker_sizes,
                  color=dur_frac,
                  colormap=:viridis,
                  colorrange=(0, maximum(dur_frac)),
                  strokewidth=0.4, strokecolor=(:black, 0.3),
                  label="XLA kernels")

    # Program-level aggregate point
    scatter!(ax, [prog_ai], [prog_gflops];
             marker=:star5, markersize=16, color=:red,
             strokewidth=1.0, strokecolor=:black,
             label=@sprintf("Aggregate (AI=%.2f)", prog_ai))

    # Label top 3 kernels by duration with leader lines
    sorted_idx = sortperm(dur_vec; rev=true)
    n_labels = min(3, length(sorted_idx))

    # Pre-defined label offsets in log-space (dx_factor, dy_factor) to avoid cluster
    # Positions are hand-tuned relative to each dot: (x_target, y_target) in data coords
    label_offsets = [
        (3.0, 400.0),  # #1 (negate_1): near yellow dot, y ~ 4e2
        (0.15, 6.0),   # #2 (add_19): left and well above
        (0.008, 0.6),  # #3 (subtract): left and below
    ]

    for j in 1:n_labels
        i = sorted_idx[j]
        short = replace(names_vec[i], r"_fusion" => "", r"^loop_" => "")
        length(short) > 20 && (short = short[1:17] * "...")
        pct = dur_frac[i] * 100
        lbl = @sprintf("%s (%.0f%%)", short, pct)

        x_dot = ai_vec[i]
        y_dot = gflops_vec[i]

        # Place label at offset position
        x_lbl, y_lbl = if j <= length(label_offsets)
            label_offsets[j]
        else
            (x_dot * 3.0, y_dot * 2.0)
        end

        # Leader line: stop just before the label text (shorten by 5% at label end)
        dx_log = log10(x_lbl) - log10(x_dot)
        dy_log = log10(y_lbl) - log10(y_dot)
        x_stop = 10^(log10(x_lbl) - 0.05 * dx_log)
        y_stop = 10^(log10(y_lbl) - 0.05 * dy_log)

        lines!(ax, [x_dot, x_stop], [y_dot, y_stop];
               color=(:gray50, 0.6), linewidth=0.7)
        # Label text
        text!(ax, x_lbl, y_lbl;
              text=lbl, fontsize=9, color=:gray20,
              align=(:left, :bottom), offset=(4, 2))
    end

    # Colorbar
    Colorbar(fig[1, 2]; colormap=:viridis,
             colorrange=(0, maximum(dur_frac) * 100),
             label="Time fraction (%)",
             labelsize=9, ticklabelsize=8, width=10)

    axislegend(ax; position=:lb, framevisible=true, labelsize=9,
               padding=(6, 6, 4, 4), rowgap=0)

    # Annotation: aggregate stats, left-aligned near ridge line
    ann_y = gf_lo * 60.0
    ann_lines = [
        @sprintf("ridge: %.1f F/B", ridge_ai),
        @sprintf("AI = %.2f F/B", prog_ai),
        @sprintf("BW = %.0f GB/s (%.0f%%)", prog_bw, prog_bw / peak_bw * 100),
        @sprintf("%d kernels", length(ai_vec)),
        @sprintf("%.1f GFLOP/ts", total_flops / 1e9),
    ]
    text!(ax, ridge_ai * 1.8, ann_y;
          text=join(ann_lines, "\n"),
          fontsize=9, color=:gray30, align=(:left, :top))

    save(output_path, fig; px_per_unit=3)
    println("Figure 1 saved: $output_path")
    return (prog_ai, prog_bw, prog_gflops, total_flops / 1e9)
end

# ── Figure 2: BW Utilization vs GPU count ─────────────────────────────

function make_bw_scaling_figure(output_path; peak_bw=PEAK_BW_GBS)
    d = XLA_DATA
    ngpus = d.ngpus
    bw    = d.xla_bw_gbs
    util  = d.bw_util_pct
    wall  = d.wall_per_ts
    n     = length(ngpus)

    # Use integer x-positions (1..n) for equal-width bars
    xs = collect(1:n)
    gpu_labels = string.(ngpus)

    # Ideal wall time (perfect weak scaling from 4 GPUs)
    ideal_wall = fill(wall[1], n)

    fig = Figure(; size=(520, 380), fontsize=11,
                  figure_padding=(4, 14, 4, 4))

    ax1 = Axis(fig[1, 1];
               xlabel="Number of GPUs",
               ylabel="HBM Bandwidth (GB/s)",
               xticks=(xs, gpu_labels),
               xlabelsize=12, ylabelsize=12,
               yticklabelcolor=:steelblue,
               ylabelcolor=:steelblue,
               xgridvisible=false,
               ygridvisible=false,
               topspinevisible=false)

    ax2 = Axis(fig[1, 1];
               ylabel="Wall time per timestep (ms)",
               yaxisposition=:right,
               xlabelsize=12, ylabelsize=12,
               yticklabelcolor=:firebrick,
               ylabelcolor=:firebrick)

    hidexdecorations!(ax2)
    hidespines!(ax2)
    ax2.xgridvisible = false
    ax2.ygridvisible = false

    # BW bars on ax1
    barplot!(ax1, xs, bw;
             color=(:steelblue, 0.15), strokewidth=1.0, strokecolor=:steelblue,
             width=0.65)

    # Peak BW reference line
    hlines!(ax1, [peak_bw]; color=(:steelblue, 0.35), linestyle=:dash, linewidth=1.0)
    text!(ax1, 0.5, peak_bw + 30;
          text=@sprintf("Peak: %.0f GB/s", peak_bw),
          fontsize=8, color=(:steelblue, 0.5), align=(:left, :bottom))

    # BW utilization as percentage labels on bars
    for i in 1:n
        text!(ax1, Float64(xs[i]), bw[i] + 30;
              text=@sprintf("%.0f%%", util[i]),
              fontsize=8, color=:steelblue, align=(:center, :bottom))
    end

    # Wall time on ax2
    lines!(ax2, Float64.(xs), wall;
           color=:firebrick, linewidth=1.8)
    scatter!(ax2, Float64.(xs), wall;
             color=:white, markersize=8,
             strokewidth=1.5, strokecolor=:firebrick)

    # Ideal wall time reference
    lines!(ax2, Float64.(xs), ideal_wall;
           color=(:firebrick, 0.35), linewidth=1.0, linestyle=:dash)

    # Label ideal line — place on ax2 (wall-time axis) so y-coordinate is in ms
    text!(ax2, Float64(n) - 0.5, wall[1] + 15;
          text=@sprintf("ideal: %.0f ms", wall[1]),
          fontsize=8, color=(:firebrick, 0.5), align=(:center, :bottom))

    # Sync axes ranges
    xlims!(ax1, 0.3, n + 0.7)
    xlims!(ax2, 0.3, n + 0.7)
    ylims!(ax1, 0, peak_bw + 250)
    ylims!(ax2, 0, maximum(wall) * 1.15)

    # Legend
    Legend(fig[2, 1],
           [PolyElement(color=(:steelblue, 0.15), strokecolor=:steelblue, strokewidth=1.0),
            [MarkerElement(marker=:circle, color=:white, markersize=8,
                           strokewidth=1.5, strokecolor=:firebrick),
             LineElement(color=:firebrick, linewidth=1.8)]],
           ["HBM bandwidth", "Wall time per timestep"],
           orientation=:horizontal, framevisible=false, labelsize=9,
           padding=(0, 0, 0, 0), rowgap=0, colgap=16)

    rowsize!(fig.layout, 2, Auto(0.08))

    save(output_path, fig; px_per_unit=3)
    println("Figure 2 saved: $output_path")
end

# ── Main ──────────────────────────────────────────────────────────────

function paper_main()
    csv_file = nothing
    output_dir = @__DIR__

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--output-dir" && i < length(ARGS)
            output_dir = ARGS[i+1]; i += 2
        elseif !startswith(ARGS[i], "-")
            csv_file = ARGS[i]; i += 1
        else
            i += 1
        end
    end

    if csv_file === nothing
        println(stderr, """
        Usage: julia --project=. plot_paper_figures.jl <ncu_csv> [--output-dir dir]

        Generates:
          fig1_roofline.png   — Single-GPU roofline scatter (ncu)
          fig2_bw_scaling.png — HBM BW utilization vs GPU count (XLA)
        """)
        exit(1)
    end

    isdir(output_dir) || mkpath(output_dir)

    # Figure 1: ncu roofline
    println("── Figure 1: Single-GPU Roofline ──")
    kernels, _ = parse_ncu_csv(csv_file)
    data = compute_roofline_data(kernels)
    agg = aggregate_roofline_data(data)
    prog_ai, prog_bw, prog_gflops, total_gflop = make_paper_roofline(
        agg, joinpath(output_dir, "fig1_roofline.png"))

    println(@sprintf("  Aggregate: AI=%.2f F/B, BW=%.0f GB/s, %.1f GFLOP/s, %.1f GFLOP/ts",
                     prog_ai, prog_bw, prog_gflops, total_gflop))

    # Figure 2: BW scaling
    println("\n── Figure 2: BW Utilization Scaling ──")
    make_bw_scaling_figure(joinpath(output_dir, "fig2_bw_scaling.png"))

end

paper_main()
