#!/usr/bin/env julia
"""
Extract XLA roofline data from xplane.pb files using Reactant.Profiler.

Parses xplane.pb from the profiling/ subdirectory of a run,
extracts program-level roofline model and per-op framework stats,
and outputs markdown tables for comparison with ncu roofline data.

Usage:
    julia --project=.. analyze_xla_roofline.jl <run_dir>
    julia --project=.. analyze_xla_roofline.jl <run_dir> --phase loop2
    julia --project=.. analyze_xla_roofline.jl <run_dir> --ngpus 4 1024
    julia --project=.. analyze_xla_roofline.jl <run_dir> --ncu-gflop 154.691 --ncu-grid 752x752x64
"""

using Reactant
using Printf

# ── Constants ─────────────────────────────────────────────────────────
const GPUS_PER_NODE = 4
const PEAK_FP64_GFLOPS = 33_500.0   # GH200 FP64 peak per GPU
const NCU_DEFAULT_GFLOP = 154.691   # ncu-measured GFLOP/timestep (single GPU)
const NCU_DEFAULT_GRID = (752, 752, 64)

# ── File discovery ────────────────────────────────────────────────────

function find_xplane(profiling_dir, job_id, rank, phase)
    phases = phase != "loop" ? [phase, "loop"] : ["loop"]
    for p in phases
        base = joinpath(profiling_dir, "$job_id.$rank", p, "plugins", "profile")
        isdir(base) || continue
        ts_dirs = sort(readdir(base))
        isempty(ts_dirs) && continue
        ts_path = joinpath(base, first(ts_dirs))
        xplanes = filter(f -> endswith(f, ".xplane.pb"), readdir(ts_path))
        !isempty(xplanes) && return joinpath(ts_path, first(xplanes)), p
    end
    return nothing, nothing
end

function parse_log(out_path)
    Nx = Ny = Nz = Ndev = nothing
    for line in eachline(out_path)
        if Nx === nothing
            m = match(r"grid:\s*(\d+)\D+(\d+)\D+(\d+)\s+LatitudeLongitude", line)
            if m !== nothing
                Nx = parse(Int, m[1])
                Ny = parse(Int, m[2])
                Nz = parse(Int, m[3])
            end
        end
        if Ndev === nothing
            m = match(r"Ndev\s*=\s*(\d+)", line)
            m !== nothing && (Ndev = parse(Int, m[1]))
        end
    end
    Nx, Ny, Nz, Ndev
end

function parse_ninner(run_dir)
    for f in readdir(run_dir)
        endswith(f, ".jl") || continue
        for line in eachline(joinpath(run_dir, f))
            m = match(r"^Ninner\s*=\s*(\d+)", line)
            m !== nothing && return parse(Int, m[1])
        end
    end
    nothing
end

function get_job_id(ngpu_dir)
    for f in readdir(ngpu_dir)
        endswith(f, ".out") && return first(split(f, "."))
    end
    nothing
end

function clip_op_name(name, maxlen=50)
    m = match(r"#([^#\"]+)#\d+", name)
    short = m !== nothing ? m[1] : name
    # Also clean up HLO names like "add.238" → "add.238"
    length(short) > maxlen ? short[1:maxlen-3] * "..." : short
end

# ── Main ──────────────────────────────────────────────────────────────

function main()
    if isempty(ARGS)
        println(stderr, """
        Usage: julia --project=.. analyze_xla_roofline.jl <run_dir> [options]
        Options:
          --phase loop2|loop   Profiling phase (default: loop2)
          --rank N             Process rank (default: 0)
          --ngpus N [N ...]    Only these GPU counts
          --ncu-gflop F        ncu GFLOP/ts reference (default: $NCU_DEFAULT_GFLOP)
          --ncu-grid NxNxN     ncu reference grid (default: $(join(NCU_DEFAULT_GRID, "x")))
          --top-ops N          Number of top ops to show (default: 15)""")
        exit(1)
    end

    run_dir = rstrip(ARGS[1], '/')
    phase = "loop2"
    rank = 0
    ngpu_filter = nothing
    ncu_gflop = NCU_DEFAULT_GFLOP
    ncu_grid = collect(NCU_DEFAULT_GRID)
    top_ops = 15

    i = 2
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--phase"
            phase = ARGS[i+1]; i += 2
        elseif arg == "--rank"
            rank = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--ngpus"
            ngpu_filter = Set{Int}()
            i += 1
            while i <= length(ARGS) && !startswith(ARGS[i], "--")
                push!(ngpu_filter, parse(Int, ARGS[i])); i += 1
            end
        elseif arg == "--ncu-gflop"
            ncu_gflop = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--ncu-grid"
            ncu_grid = [parse(Int, x) for x in split(ARGS[i+1], "x")]; i += 2
        elseif arg == "--top-ops"
            top_ops = parse(Int, ARGS[i+1]); i += 2
        else
            i += 1
        end
    end

    ncu_cells = prod(ncu_grid)
    profiling_dir = joinpath(run_dir, "profiling")
    run_name = basename(run_dir)

    ninner = something(parse_ninner(run_dir), 256)
    ninner == 256 && !any(endswith(".jl"), readdir(run_dir)) &&
        println(stderr, "WARNING: Ninner not found, using default 256")

    ngpu_dirs = sort(filter(d -> startswith(d, "ngpu="), readdir(run_dir)))
    if ngpu_filter !== nothing
        ngpu_dirs = filter(d -> parse(Int, split(d, "=")[2]) ∈ ngpu_filter, ngpu_dirs)
    end

    results = []
    all_op_stats = Dict{Int,Vector}()

    for ngpu_name in ngpu_dirs
        ngpu = parse(Int, split(ngpu_name, "=")[2])
        ngpu_path = joinpath(run_dir, ngpu_name)

        job_id = get_job_id(ngpu_path)
        if job_id === nothing
            println(stderr, "  SKIP ngpu=$ngpu: no .out file"); continue
        end

        Nx, Ny, Nz, Ndev = parse_log(joinpath(ngpu_path, "$job_id.out"))
        if Nx === nothing
            println(stderr, "  SKIP ngpu=$ngpu: no grid info"); continue
        end

        per_gpu_cells = Nx * Ny * Nz / Ndev
        cell_ratio = per_gpu_cells / ncu_cells
        ncu_scaled = ncu_gflop * cell_ratio

        xplane_path, actual_phase = find_xplane(profiling_dir, job_id, rank, phase)
        if xplane_path === nothing
            println(stderr, "  SKIP ngpu=$ngpu: no xplane.pb"); continue
        end

        println(stderr, "  Parsing ngpu=$ngpu ($actual_phase)...")

        # Extract program-level roofline
        roofline = Reactant.Profiler.get_total_program_roofline(xplane_path)
        if isempty(roofline)
            println(stderr, "  SKIP ngpu=$ngpu: roofline extraction failed"); continue
        end

        # Extract per-op framework stats
        ops = try
            Reactant.Profiler.get_framework_op_stats(xplane_path)
        catch e
            println(stderr, "  WARNING ngpu=$ngpu: op stats failed: $e")
            []
        end
        !isempty(ops) && (all_op_stats[ngpu] = ops)

        nnodes = cld(ngpu, GPUS_PER_NODE)

        # XLA roofline metrics (per-GPU averages)
        total_time_per_core_us = Float64(get(roofline, "total_time_per_core", 0.0))
        wall_per_ts_ms = total_time_per_core_us / ninner / 1e3

        xla_flop_rate   = Float64(get(roofline, "measured_flop_rate", 0.0))   # GFLOP/s (model FLOPs / measured time)
        xla_mem_bw      = Float64(get(roofline, "measured_memory_bw", 0.0))   # GB/s
        xla_hbm_bw      = Float64(get(roofline, "hbm_bw", 0.0))             # GB/s
        xla_intensity   = Float64(get(roofline, "operational_intensity", 0.0))# FLOP/byte
        xla_efficiency  = Float64(get(roofline, "roofline_efficiency", 0.0))
        xla_bound       = String(get(roofline, "bound_by", "?"))
        xla_bw_util     = Float64(get(roofline, "max_mem_bw_utilization", 0.0))
        xla_compute_eff = Float64(get(roofline, "compute_efficiency", 0.0))

        # ncu-based throughput using XLA wall time
        ncu_node_gflops = wall_per_ts_ms > 0 ?
            ncu_scaled * GPUS_PER_NODE / (wall_per_ts_ms / 1e3) : 0.0
        ncu_agg_pflops = wall_per_ts_ms > 0 ?
            ncu_scaled * ngpu / (wall_per_ts_ms / 1e3) / 1e6 : 0.0
        ncu_fp64_pct = wall_per_ts_ms > 0 ?
            ncu_node_gflops / (GPUS_PER_NODE * PEAK_FP64_GFLOPS) * 100 : 0.0

        push!(results, (;
            ngpu, nnodes, per_gpu_cells, cell_ratio, ncu_scaled,
            wall_per_ts_ms, xla_flop_rate, xla_mem_bw, xla_hbm_bw,
            xla_intensity, xla_efficiency, xla_bound, xla_bw_util,
            xla_compute_eff, ncu_node_gflops, ncu_agg_pflops, ncu_fp64_pct,
            phase=actual_phase,
        ))
    end

    isempty(results) && (println("No results."); return)

    # ── Output ────────────────────────────────────────────────────────
    println()
    println("# XLA Roofline Analysis: $run_name")
    println()
    println("- **ncu reference**: $(@sprintf("%.3f", ncu_gflop)) GFLOP/ts on $(join(ncu_grid, "×")) grid")
    println("- **Ninner**: $ninner")
    println("- **Phase**: $phase (rank $rank)")
    println("- **GH200 peak**: FP64 $(@sprintf("%.0f", PEAK_FP64_GFLOPS)) GFLOP/s/GPU")
    println()

    # ── Program-level roofline table ──────────────────────────────────
    println("## Program-Level Roofline (per-GPU averages)")
    println()
    println("| GPUs | Nodes | XLA FLOP Rate | XLA BW (GB/s) | BW Util | Intensity (F/B) | Roofline Eff | Bound | ncu Node GFLOP/s | ncu FP64 % | Wall/ts (ms) |")
    println("|-----:|------:|--------------:|--------------:|--------:|----------------:|-------------:|------:|-----------------:|-----------:|-------------:|")

    for r in results
        ncu_str = r.ncu_node_gflops > 0 ? @sprintf("%.0f", r.ncu_node_gflops) : "N/A"
        @printf("| %4d | %5d | %13.0f | %13.1f | %6.1f%% | %15.1f | %11.1f%% | %5s | %16s | %9.1f%% | %12.2f |\n",
            r.ngpu, r.nnodes,
            r.xla_flop_rate, r.xla_mem_bw, r.xla_bw_util * 100,
            r.xla_intensity, r.xla_efficiency * 100, r.xla_bound,
            ncu_str, r.ncu_fp64_pct, r.wall_per_ts_ms)
    end

    # ── Scaling summary ───────────────────────────────────────────────
    if length(results) > 1
        println()
        println("## Scaling Summary")
        println()
        println("| GPUs | Nodes | ncu GFLOP/ts | ncu Agg PFLOP/s | ncu FP64 % | XLA Roofline Eff | XLA Bound |")
        println("|-----:|------:|-------------:|----------------:|-----------:|-----------------:|----------:|")
        for r in results
            agg_str = r.ncu_agg_pflops > 0 ? @sprintf("%.4f", r.ncu_agg_pflops) : "N/A"
            @printf("| %4d | %5d | %12.1f | %15s | %9.1f%% | %15.1f%% | %9s |\n",
                r.ngpu, r.nnodes, r.ncu_scaled, agg_str,
                r.ncu_fp64_pct, r.xla_efficiency * 100, r.xla_bound)
        end
    end

    # ── Per-op breakdown ──────────────────────────────────────────────
    println()
    println("## Top Framework Ops by Self-Time")
    println()

    show_ngpus = unique([first(results).ngpu, last(results).ngpu])
    for ngpu in show_ngpus
        haskey(all_op_stats, ngpu) || continue
        ops = sort(all_op_stats[ngpu]; by=r -> r.total_self_time_in_us, rev=true)
        n = min(top_ops, length(ops))

        # Compute total device time for percentage
        total_self_us = sum(op.total_self_time_in_us for op in ops)

        println("### $ngpu GPUs ($(cld(ngpu, GPUS_PER_NODE)) nodes)")
        println()
        println("| # | Operation | Occ | Self-Time (s) | Time % | FLOP Rate | BW (GB/s) | Intensity | Bound |")
        println("|--:|-----------|----:|--------------:|-------:|----------:|----------:|----------:|------:|")

        cumulative_pct = 0.0
        for (i, op) in enumerate(ops[1:n])
            name = clip_op_name(op.op_name)
            self_s = op.total_self_time_in_us / 1e6
            pct = total_self_us > 0 ? op.total_self_time_in_us / total_self_us * 100 : 0.0
            cumulative_pct += pct
            @printf("| %2d | %-50s | %5d | %13.3f | %5.1f%% | %9.0f | %9.1f | %9.1f | %5s |\n",
                i, name, op.occurrences, self_s,
                pct, op.model_flop_rate, op.measured_memory_bw,
                op.operational_intensity, op.bound_by)
        end
        @printf("|    | %-50s |       | %13.3f | %5.1f%% |           |           |           |       |\n",
            "**Top $n total**", sum(ops[j].total_self_time_in_us for j in 1:n) / 1e6, cumulative_pct)
        println()
    end

    # ── Bound classification summary ──────────────────────────────────
    println("## Bound Classification Summary")
    println()
    for ngpu in show_ngpus
        haskey(all_op_stats, ngpu) || continue
        ops = all_op_stats[ngpu]
        total_us = sum(op.total_self_time_in_us for op in ops)

        bound_time = Dict{String,Float64}()
        for op in ops
            b = op.bound_by
            bound_time[b] = get(bound_time, b, 0.0) + op.total_self_time_in_us
        end

        println("**$ngpu GPUs**: ", join(
            ["$b: $(@sprintf("%.1f", v/total_us*100))%" for (b, v) in sort(collect(bound_time); by=last, rev=true)],
            ", "))
    end

    # ── Notes ─────────────────────────────────────────────────────────
    println()
    println("## Methodology")
    println()
    println("- **XLA FLOP Rate**: HLO cost-model FLOPs ÷ measured GPU time (counts all op types, not just FP64)")
    println("- **XLA BW**: Memory bandwidth from XLA device counters (per-GPU average)")
    println("- **BW Util**: Fraction of GH200 HBM3 peak bandwidth")
    println("- **Intensity**: XLA model FLOPs ÷ measured bytes moved")
    println("- **Roofline Eff**: Measured rate ÷ roofline ceiling at measured intensity")
    println("- **ncu Node GFLOP/s**: ncu HW-measured FP64 FLOPs × cell ratio × 4 GPUs ÷ XLA wall time")
    println("- **ncu FP64 %**: ncu Node GFLOP/s ÷ (4 × 33500) = fraction of FP64 peak")
    println("- XLA model FLOPs >> ncu FP64 FLOPs (XLA counts selects, broadcasts, integer ops, etc.)")
    println("- Memory BW and wall time are directly comparable between XLA and ncu")
end

main()
