#=
Analyze FLOP counts from an ncu CSV export (--csv --page raw).

Reads the wide-format CSV produced by:
    ncu -i <report>.ncu-rep --csv --page raw > raw.csv

Computes per-kernel and total FLOPs breakdown (FP64, FP32, FP16),
effective GFLOP/s, and arithmetic intensity (if DRAM bytes available).

Usage:
    julia analyze_ncu_flops.jl <raw.csv>
    julia analyze_ncu_flops.jl <raw.csv> --top 20
    julia analyze_ncu_flops.jl <raw.csv> --per-kernel
=#

using Printf

# ── Metric column names ──────────────────────────────────────────────
# Format A: --metrics (raw .sum instruction counts)
const DADD = "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"
const DMUL = "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"
const DFMA = "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"

const FADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
const FMUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
const FFMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"

const HADD = "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"
const HMUL = "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"
const HFMA = "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"

# Format B: --set roofline (derived metrics, pre-scaled for FLOPs)
# _x2 = already multiplied by 2 for FMA (2 FLOPs per instruction)
# _x4 = already multiplied by 2 (FMA) × 2 (half-precision 2-wide)
const DFMA_X2  = "derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2"
const FFMA_X2  = "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2"
const HFMA_X4  = "derived__sm__sass_thread_inst_executed_op_hfma_pred_on_x4"
const HADD_X2  = "derived__smsp__sass_thread_inst_executed_op_hadd_pred_on_x2"
const HMUL_X2  = "derived__smsp__sass_thread_inst_executed_op_hmul_pred_on_x2"
# Per-cycle-elapsed variants (need cycles to convert to absolute counts)
const DADD_PER_CYCLE = "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed"
const DMUL_PER_CYCLE = "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed"
const DFMA_PER_CYCLE = "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed"
const FADD_PER_CYCLE = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"
const FMUL_PER_CYCLE = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"
const HADD_PER_CYCLE = "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed"
const HMUL_PER_CYCLE = "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed"
# Cycles elapsed (to convert per_cycle_elapsed to absolute)
const SM_CYCLES = "sm__cycles_elapsed.avg"
const SMSP_CYCLES = "smsp__cycles_elapsed.avg"

const DURATION = "gpu__time_duration.sum"
const DRAM_BYTES = "dram__bytes.sum"
const DRAM_BW = "dram__bytes.sum.per_second"          # Tbyte/s (roofline set)
const DRAM_BW_PEAK = "dram__bytes.sum.peak_sustained"  # Kbyte/cycle (roofline set)

const ALL_METRICS_A = [DADD, DMUL, DFMA, FADD, FMUL, FFMA, HADD, HMUL, HFMA, DURATION, DRAM_BYTES]
const ALL_METRICS_B = [DFMA_X2, FFMA_X2, HFMA_X4, HADD_X2, HMUL_X2,
                        DADD_PER_CYCLE, DMUL_PER_CYCLE, DFMA_PER_CYCLE,
                        FADD_PER_CYCLE, FMUL_PER_CYCLE, HADD_PER_CYCLE, HMUL_PER_CYCLE,
                        SM_CYCLES, SMSP_CYCLES, DURATION, DRAM_BW, DRAM_BW_PEAK]

struct KernelData
    name::String
    duration_ms::Float64
    fp64_flops::Float64
    fp32_flops::Float64
    fp16_flops::Float64
    total_flops::Float64
    dram_bytes::Float64
end

function safe_parse(val::AbstractString)
    s = strip(replace(val, "," => ""))
    (isempty(s) || s == "\"\"" || lowercase(s) == "no data" || s == "n/a") && return 0.0
    return parse(Float64, s)
end

function parse_ncu_csv(csv_path::String)
    lines = readlines(csv_path)
    length(lines) < 3 && error("CSV must have at least header + units + 1 data row")

    # Parse header (line 1) — CSV with quoted fields
    header = split_csv_line(lines[1])
    # Skip units row (line 2)

    col = Dict(name => idx for (idx, name) in enumerate(header))

    # Detect format: A (raw .sum) vs B (roofline derived)
    has_raw_sum = haskey(col, DADD) || haskey(col, DFMA)
    has_derived = haskey(col, DFMA_X2) || haskey(col, DADD_PER_CYCLE)

    all_metrics = has_raw_sum ? ALL_METRICS_A : ALL_METRICS_B
    available = Set(m for m in all_metrics if haskey(col, m))
    format = has_raw_sum ? :raw : :roofline

    get_val(fields, metric) = haskey(col, metric) ? safe_parse(fields[col[metric]]) : 0.0

    kernels = KernelData[]
    for i in 3:length(lines)
        isempty(strip(lines[i])) && continue
        fields = split_csv_line(lines[i])
        length(fields) < length(header) && continue

        name = haskey(col, "Kernel Name") ? strip(fields[col["Kernel Name"]], '"') : "unknown"
        duration_ms = get_val(fields, DURATION)

        if format == :raw
            # Format A: raw instruction counts
            dadd = get_val(fields, DADD)
            dmul = get_val(fields, DMUL)
            dfma = get_val(fields, DFMA)
            fadd = get_val(fields, FADD)
            fmul = get_val(fields, FMUL)
            ffma = get_val(fields, FFMA)
            hadd = get_val(fields, HADD)
            hmul = get_val(fields, HMUL)
            hfma = get_val(fields, HFMA)

            fp64_flops = dadd + dmul + 2 * dfma
            fp32_flops = fadd + fmul + 2 * ffma
            fp16_flops = hadd + hmul + 2 * hfma

            dram_bytes_raw = get_val(fields, DRAM_BYTES)
            dram_bytes_actual = DRAM_BYTES in available ? dram_bytes_raw * 1e9 : 0.0
        else
            # Format B: roofline derived metrics
            # Get cycle count to convert per_cycle_elapsed to absolute
            sm_cyc = get_val(fields, SM_CYCLES)
            smsp_cyc = get_val(fields, SMSP_CYCLES)
            # Use smsp cycles for smsp metrics, sm cycles for sm metrics
            cyc = smsp_cyc > 0 ? smsp_cyc : sm_cyc

            # FP64: derived dfma_x2 already = dfma_flops, plus dadd + dmul from per_cycle
            dfma_flops = get_val(fields, DFMA_X2)  # already ×2
            dadd_abs = get_val(fields, DADD_PER_CYCLE) * cyc
            dmul_abs = get_val(fields, DMUL_PER_CYCLE) * cyc
            fp64_flops = dfma_flops + dadd_abs + dmul_abs

            # FP32: derived ffma_x2 + fadd + fmul from per_cycle
            ffma_flops = get_val(fields, FFMA_X2)  # already ×2
            fadd_abs = get_val(fields, FADD_PER_CYCLE) * cyc
            fmul_abs = get_val(fields, FMUL_PER_CYCLE) * cyc
            fp32_flops = ffma_flops + fadd_abs + fmul_abs

            # FP16: derived hfma_x4 + hadd_x2 + hmul_x2
            hfma_flops = get_val(fields, HFMA_X4)    # already ×4 (2-wide × FMA)
            hadd_flops = get_val(fields, HADD_X2)     # already ×2 (2-wide)
            hmul_flops = get_val(fields, HMUL_X2)     # already ×2 (2-wide)
            # Also try per_cycle if derived not available
            if hfma_flops == 0 && hadd_flops == 0 && hmul_flops == 0
                hadd_flops = get_val(fields, HADD_PER_CYCLE) * cyc
                hmul_flops = get_val(fields, HMUL_PER_CYCLE) * cyc
            end
            fp16_flops = hfma_flops + hadd_flops + hmul_flops

            # DRAM bytes: BW (Tbyte/s) × duration (ms → s) = Tbyte → bytes
            dram_bw_tbs = get_val(fields, DRAM_BW)  # Tbyte/s
            dram_bytes_actual = dram_bw_tbs * (duration_ms / 1e3) * 1e12  # convert to bytes
        end

        push!(kernels, KernelData(name, duration_ms, fp64_flops, fp32_flops, fp16_flops,
                                   fp64_flops + fp32_flops + fp16_flops, dram_bytes_actual))
    end
    return kernels, available
end

function split_csv_line(line::AbstractString)
    fields = String[]
    current = IOBuffer()
    in_quotes = false
    for c in line
        if c == '"'
            in_quotes = !in_quotes
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(current)))
        else
            write(current, c)
        end
    end
    push!(fields, String(take!(current)))
    return fields
end

function commas(n::Real)
    s = string(round(Int, n))
    # Insert commas from the right
    parts = String[]
    while length(s) > 3
        push!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    push!(parts, s)
    return join(reverse(parts), ",")
end

function fmt_gflop(flops)
    return @sprintf("%.3f", flops / 1e9)
end

function print_markdown(kernels, available; top_n=nothing, per_kernel=false)
    total_fp64 = sum(k.fp64_flops for k in kernels)
    total_fp32 = sum(k.fp32_flops for k in kernels)
    total_fp16 = sum(k.fp16_flops for k in kernels)
    total_flops = total_fp64 + total_fp32 + total_fp16
    total_duration_ms = sum(k.duration_ms for k in kernels)
    total_dram = sum(k.dram_bytes for k in kernels)
    unique_kernels = length(Set(k.name for k in kernels))

    println("# NCU FLOPs Analysis\n")

    println("## Overview\n")
    println("| Metric | Value |")
    println("|--------|-------|")
    println("| Kernel launches profiled | $(length(kernels)) |")
    println("| Unique kernels | $unique_kernels |")
    @printf("| Total GPU time | %.3f ms |\n", total_duration_ms)
    println()

    println("## FLOP Counts\n")
    println("| Precision | Instructions | GFLOP | Share |")
    println("|-----------|-------------|-------|-------|")
    for (label, val) in [("FP64", total_fp64), ("FP32", total_fp32), ("FP16", total_fp16)]
        pct = total_flops > 0 ? 100 * val / total_flops : 0.0
        @printf("| %s | %s | %.3f | %.2f%% |\n", label, commas(val), val / 1e9, pct)
    end
    @printf("| **Total** | **%s** | **%.3f** | **100%%** |\n", commas(total_flops), total_flops / 1e9)
    println()

    if total_duration_ms > 0
        gflops_per_sec = (total_flops / 1e9) / (total_duration_ms / 1e3)
        println("## Effective Throughput\n")
        println("| Metric | Value |")
        println("|--------|-------|")
        @printf("| Total | %.1f GFLOP/s |\n", gflops_per_sec)
        if total_fp64 > 0
            fp64_gflops = (total_fp64 / 1e9) / (total_duration_ms / 1e3)
            @printf("| FP64 only | %.1f GFLOP/s |\n", fp64_gflops)
        end
        println()
    end

    if total_dram > 0 && total_flops > 0
        ai = total_flops / total_dram
        println("## Arithmetic Intensity\n")
        @printf("**%.3f FLOP/byte**\n\n", ai)
    end

    # Aggregate by kernel name
    agg = Dict{String,NamedTuple{(:fp64,:fp32,:fp16,:total,:duration_ms,:count,:dram_bytes),
                                  NTuple{7,Float64}}}()
    for k in kernels
        prev = get(agg, k.name, (fp64=0.0, fp32=0.0, fp16=0.0, total=0.0,
                                   duration_ms=0.0, count=0.0, dram_bytes=0.0))
        agg[k.name] = (fp64      = prev.fp64 + k.fp64_flops,
                        fp32      = prev.fp32 + k.fp32_flops,
                        fp16      = prev.fp16 + k.fp16_flops,
                        total     = prev.total + k.total_flops,
                        duration_ms = prev.duration_ms + k.duration_ms,
                        count     = prev.count + 1,
                        dram_bytes = prev.dram_bytes + k.dram_bytes)
    end

    sorted_kernels = sort(collect(agg); by=x -> x.second.total, rev=true)

    n = something(top_n, per_kernel ? length(sorted_kernels) : 20)
    n = min(n, length(sorted_kernels))

    println("## Top $n Kernels by FLOP Count\n")
    println("| # | Launches | Duration (ms) | GFLOP | GFLOP/s | Kernel Name |")
    println("|--:|--------:|--------------:|------:|--------:|-------------|")
    for (i, (name, a)) in enumerate(sorted_kernels[1:n])
        gf = a.total / 1e9
        gfs = a.duration_ms > 0 ? gf / (a.duration_ms / 1e3) : 0.0
        @printf("| %d | %d | %.3f | %.3f | %.1f | `%s` |\n",
                i, Int(a.count), a.duration_ms, gf, gfs, name)
    end
    gflops_total = total_flops / 1e9
    gflops_s_total = total_duration_ms > 0 ? gflops_total / (total_duration_ms / 1e3) : 0.0
    @printf("| | **%d** | **%.3f** | **%.3f** | **%.1f** | **Total** |\n",
            length(kernels), total_duration_ms, gflops_total, gflops_s_total)
    println()
end

function main()
    args = ARGS
    csv_file = nothing
    top_n = nothing
    per_kernel = false

    i = 1
    while i <= length(args)
        if args[i] == "--top" && i < length(args)
            top_n = parse(Int, args[i+1])
            i += 2
        elseif args[i] == "--per-kernel"
            per_kernel = true
            i += 1
        elseif startswith(args[i], "-")
            error("Unknown flag: $(args[i])")
            i += 1
        else
            csv_file = args[i]
            i += 1
        end
    end

    if csv_file === nothing
        println(stderr, "Usage: julia analyze_ncu_flops.jl <raw.csv> [--top N] [--per-kernel]")
        exit(1)
    end

    if !isfile(csv_file)
        println(stderr, "ERROR: File not found: $csv_file")
        exit(1)
    end

    kernels, available = parse_ncu_csv(csv_file)
    if isempty(kernels)
        println(stderr, "No kernel data found in CSV.")
        exit(1)
    end

    println("*Metrics available: $(join(sort(collect(available)), ", "))*\n")
    print_markdown(kernels, available; top_n, per_kernel)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
