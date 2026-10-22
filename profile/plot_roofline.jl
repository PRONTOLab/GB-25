#=
Generate a roofline plot (PNG) from an ncu CSV export (--csv --page raw, from --set roofline).

Each kernel is plotted as a point at (arithmetic_intensity, achieved_GFLOP/s).
The roofline ceiling is drawn from the device peak bandwidth and compute.

Usage:
    julia --project=. plot_roofline.jl <raw.csv>
    julia --project=. plot_roofline.jl <raw.csv> --output roofline.png
    julia --project=. plot_roofline.jl <raw.csv> --top 20

Requires CairoMakie (in project deps).
=#

using Printf
using CairoMakie

# Include the CSV parser from analyze_ncu_flops.jl
include(joinpath(@__DIR__, "analyze_ncu_flops.jl"))

# ── GH200 120GB peak specs ───────────────────────────────────────────
const PEAK_FP64_GFLOPS = 33_500.0  # ~33.5 TFLOP/s FP64 (official NVIDIA spec)
const PEAK_FP32_GFLOPS = 67_000.0  # ~67 TFLOP/s FP32
const PEAK_DRAM_GBS    = 4_000.0   # ~4.0 TB/s HBM3

function compute_roofline_data(kernels)
    data = NamedTuple{(:name, :ai, :gflops, :duration_ms, :total_flops, :dram_bytes),
                       Tuple{String, Float64, Float64, Float64, Float64, Float64}}[]
    for k in kernels
        k.duration_ms <= 0 && continue
        k.total_flops <= 0 && k.dram_bytes <= 0 && continue

        gflops = (k.total_flops / 1e9) / (k.duration_ms / 1e3)
        ai = k.dram_bytes > 0 ? k.total_flops / k.dram_bytes : Inf
        (isnan(ai) || isinf(ai) || ai <= 0) && continue
        (isnan(gflops) || gflops <= 0) && continue

        push!(data, (name=k.name, ai=ai, gflops=gflops, duration_ms=k.duration_ms,
                      total_flops=k.total_flops, dram_bytes=k.dram_bytes))
    end
    return data
end

function aggregate_roofline_data(data)
    agg = Dict{String, NamedTuple{(:ai, :gflops, :duration_ms, :total_flops, :dram_bytes, :count),
                                    NTuple{6, Float64}}}()
    for d in data
        prev = get(agg, d.name, (ai=0.0, gflops=0.0, duration_ms=0.0,
                                   total_flops=0.0, dram_bytes=0.0, count=0.0))
        new_flops = prev.total_flops + d.total_flops
        new_bytes = prev.dram_bytes + d.dram_bytes
        new_dur   = prev.duration_ms + d.duration_ms
        new_ai    = new_bytes > 0 ? new_flops / new_bytes : 0.0
        new_gflops = new_dur > 0 ? (new_flops / 1e9) / (new_dur / 1e3) : 0.0
        agg[d.name] = (ai=new_ai, gflops=new_gflops, duration_ms=new_dur,
                        total_flops=new_flops, dram_bytes=new_bytes, count=prev.count + 1)
    end
    return agg
end

function print_roofline_table(agg; top_n=20)
    sorted = sort(collect(agg); by=x -> x.second.total_flops, rev=true)
    n = min(top_n, length(sorted))

    println("\n## Roofline Data (Top $n Kernels by FLOP count)\n")
    println("| # | Launches | AI (FLOP/B) | GFLOP/s | Duration (ms) | GFLOP | GB xfer | Kernel |")
    println("|--:|--------:|------------:|--------:|--------------:|------:|--------:|--------|")
    for (i, (name, a)) in enumerate(sorted[1:n])
        gb = a.dram_bytes / 1e9
        gf = a.total_flops / 1e9
        @printf("| %d | %d | %.3f | %.1f | %.3f | %.3f | %.2f | `%s` |\n",
                i, Int(a.count), a.ai, a.gflops, a.duration_ms, gf, gb, name)
    end
    println()
end

# ── CairoMakie roofline plot ─────────────────────────────────────────

function make_roofline_plot(agg, output_path;
                             peak_fp64=PEAK_FP64_GFLOPS, peak_bw=PEAK_DRAM_GBS)
    # Collect data
    names_vec = String[]
    ai_vec = Float64[]
    gflops_vec = Float64[]
    flops_vec = Float64[]
    for (name, a) in agg
        a.ai > 0 && a.gflops > 0 || continue
        push!(names_vec, name)
        push!(ai_vec, a.ai)
        push!(gflops_vec, a.gflops)
        push!(flops_vec, a.total_flops)
    end

    isempty(ai_vec) && error("No valid roofline data points")

    # Ridge point: where BW ceiling meets compute ceiling
    ridge_ai = peak_fp64 / peak_bw

    # Roofline ceiling lines
    ai_lo = 10.0^(floor(log10(minimum(ai_vec))) - 0.5)
    ai_hi = 10.0^(ceil(log10(maximum(ai_vec))) + 0.5)
    # BW-limited segment: from ai_lo to ridge
    bw_ai = [ai_lo, ridge_ai]
    bw_gf = peak_bw .* bw_ai
    # Compute-limited segment: from ridge to ai_hi
    cp_ai = [ridge_ai, ai_hi]
    cp_gf = [peak_fp64, peak_fp64]

    # Marker sizes proportional to log(FLOP count)
    log_flops = log10.(max.(flops_vec, 1.0))
    min_lf, max_lf = extrema(log_flops)
    marker_sizes = if max_lf > min_lf
        8.0 .+ 18.0 .* (log_flops .- min_lf) ./ (max_lf - min_lf)
    else
        fill(12.0, length(log_flops))
    end

    fig = Figure(; size=(1000, 650), fontsize=14)
    ax = Axis(fig[1, 1];
              xlabel="Arithmetic Intensity (FLOP/byte)",
              ylabel="Performance (GFLOP/s)",
              title="Roofline Analysis — NVIDIA GH200 120GB (single GPU)",
              xscale=log10, yscale=log10,
              xminorticksvisible=true, yminorticksvisible=true,
              xminorticks=IntervalsBetween(9),
              yminorticks=IntervalsBetween(9))

    # Draw roofline ceiling
    lines!(ax, bw_ai, bw_gf; color=:red, linewidth=2.5,
           label=@sprintf("Peak BW: %.0f GB/s", peak_bw))
    lines!(ax, cp_ai, cp_gf; color=:dodgerblue, linewidth=2.5,
           label=@sprintf("Peak FP64: %.1f TFLOP/s", peak_fp64 / 1000))

    # Ridge point vertical line
    vlines!(ax, [ridge_ai]; color=(:gray60, 0.6), linestyle=:dash, linewidth=1)
    text!(ax, ridge_ai * 1.15, minimum(gflops_vec) * 1.5;
          text=@sprintf("ridge: %.1f F/B", ridge_ai),
          fontsize=10, color=:gray40)

    # Kernel scatter
    scatter!(ax, ai_vec, gflops_vec;
             markersize=marker_sizes,
             color=(:royalblue, 0.5),
             strokewidth=0.5, strokecolor=(:black, 0.3),
             label="Kernels (size ∝ log FLOP)")

    # Label top 5 kernels by GFLOP/s
    sorted_idx = sortperm(gflops_vec; rev=true)
    for j in 1:min(5, length(sorted_idx))
        i = sorted_idx[j]
        short = replace(names_vec[i], r"_fusion" => "")
        if length(short) > 35
            short = short[1:32] * "..."
        end
        text!(ax, ai_vec[i], gflops_vec[i];
              text=short, fontsize=8, align=(:left, :bottom), offset=(6, 4))
    end

    axislegend(ax; position=:lb, framevisible=true, labelsize=11, padding=(8, 8, 6, 6))

    save(output_path, fig; px_per_unit=2)
    println("Roofline plot saved to: $output_path")
end

function roofline_main()
    args = ARGS
    csv_file = nothing
    output = nothing
    top_n = 20

    i = 1
    while i <= length(args)
        if args[i] == "--output" && i < length(args)
            output = args[i+1]
            i += 2
        elseif args[i] == "--top" && i < length(args)
            top_n = parse(Int, args[i+1])
            i += 2
        elseif !startswith(args[i], "-")
            csv_file = args[i]
            i += 1
        else
            i += 1
        end
    end

    if csv_file === nothing
        println(stderr, "Usage: julia --project=. plot_roofline.jl <raw.csv> [--output file.png] [--top N]")
        exit(1)
    end

    kernels, available = parse_ncu_csv(csv_file)
    isempty(kernels) && error("No kernel data found")

    data = compute_roofline_data(kernels)
    isempty(data) && error("No kernels with both FLOP and DRAM data for roofline")

    agg = aggregate_roofline_data(data)

    # Print roofline table
    print_roofline_table(agg; top_n)

    # Generate PNG plot
    if output === nothing
        output = replace(csv_file, ".csv" => "_roofline.png")
    end
    make_roofline_plot(agg, output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    roofline_main()
end
