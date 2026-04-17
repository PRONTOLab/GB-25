const DEFAULT_LABEL = "second loop"

"""
Extract all timings for the given label from a log file.
"""
function extract_timings(path::AbstractString, label::AbstractString)
    # Match lines like: [15] second loop: 7.192586 seconds
    pattern = Regex(raw"^\[(\d+)\]\s+" * escape_string(label) * raw":\s+([0-9]+(?:\.[0-9]+)?)\s+seconds\b")
    timings = NamedTuple{(:rank, :seconds, :line)}[]

    open(path, "r") do io
        for (lineno, line) in enumerate(eachline(io))
            match_result = match(pattern, line)
            match_result === nothing && continue

            push!(timings, (
                rank = parse(Int, match_result.captures[1]),
                seconds = parse(Float64, match_result.captures[2]),
                line = lineno,
            ))
        end
    end

    isempty(timings) && @warn "No timings found for label \"$(label)\" in $(path)"

    return timings
end

"""
Extract the GPU count from the sibling submit.sh srun command.
"""
function extract_gpu_count(path::AbstractString)
    submit_path = joinpath(dirname(path), "submit.sh")
    isfile(submit_path) || error("Could not find submit.sh next to $(path)")

    contents = read(submit_path, String)
    # Require exactly one srun invocation so the GPU count is unambiguous.
    srun_matches = collect(eachmatch(r"(?m)^\s*(srun\b[^\n]*)", contents))
    isempty(srun_matches) && error("No line starting with \"srun\" found in $(submit_path)")
    length(srun_matches) == 1 || error("Expected exactly one line starting with \"srun\" in $(submit_path), found $(length(srun_matches))")

    srun_line = srun_matches[1].captures[1]
    # Accept both "-G 3872" and "-G=3872".
    gpu_match = match(r"(?:^|\s)-G(?:\s*|=)([0-9]+)\b", srun_line)
    gpu_match === nothing && error("Found srun line in $(submit_path), but no -G flag was present")

    return parse(Int, gpu_match.captures[1])
end

"""
Find the unique .out file in a run directory.
"""
function find_log_file(run_dir::AbstractString)
    out_files = filter(name -> endswith(name, ".out"), readdir(run_dir; join=true))
    isempty(out_files) && @warn "No .out files found in $(run_dir)"
    length(out_files) == 1 || error("Expected exactly one .out file in $(run_dir), found $(length(out_files))")

    return out_files[1]
end

"""
Run the parser for each ngpu* subdirectory under a weak-scaling run directory.
"""
function slurp(path::AbstractString, label::AbstractString=DEFAULT_LABEL)
    run_dirs = sort(filter(entry -> isdir(entry) && startswith(basename(entry), "ngpu"), readdir(path; join=true)))
    isempty(run_dirs) && error("No ngpu* subdirectories found in $(path)")
    results = Matrix{Float64}(undef, 0, 2)

    for run_dir in run_dirs
        out_files = filter(name -> endswith(name, ".out"), readdir(run_dir; join=true))
        isempty(out_files) && continue

        log_path = find_log_file(run_dir)
        timings = extract_timings(log_path, label)
        isempty(timings) && continue
        gpu_count = extract_gpu_count(log_path)

        # sanity check
        if length(timings) * 4 != gpu_count
            @warn "Expected gpu_count == matches * 4, but got gpus=$(gpu_count) and matches=$(length(timings))"
            continue
        end

        # get min of timings - apparently this is the right thing to do for weak scaling
        # in this case
        best = reduce((a, b) -> a.seconds <= b.seconds ? a : b, timings)

        println("$(basename(run_dir)) $(label): min_seconds=$(best.seconds), ngpu=$(gpu_count)")
        results = vcat(results, reshape([Float64(gpu_count), best.seconds], 1, 2))
    end

    return results
end

"""
Run slurp over a list of weak-scaling parent directories and combine the results.
"""
function slurp_all(paths::AbstractVector{<:AbstractString}, label::AbstractString=DEFAULT_LABEL)
    results = Matrix{Float64}(undef, 0, 2)

    for path in paths
        println("path=$(basename(path))")
        results = vcat(results, slurp(path, label))
    end

    return results
end


# ----------------------------------
# analyze
# ----------------------------------

paths = [
    "/global/cfs/cdirs/m5176/roman/GB25/2026.04.11.gb_re_26_1/ocean/scaling/comm_opts/2026-04-11T17-00-24.828_ngpu3872",
    "/global/cfs/cdirs/m5176/roman/GB25/2026.04.11.gb_re_26_1/ocean/scaling/comm_opts/2026-04-11T17-28-57.416_ngpu6136",
]

co = slurp_all(paths)
display(co)

paths = [
    "/global/cfs/cdirs/m5176/roman/GB25/2026.04.11.gb_re_26_1/ocean/scaling/ablation/2026-04-11T16-08-21.868_ngpu2048"
    "/global/cfs/cdirs/m5176/roman/GB25/2026.04.11.gb_re_26_1/ocean/scaling/ablation/2026-04-11T16-25-49.409_ablation_ngpu3872"
    "/global/cfs/cdirs/m5176/roman/GB25/2026.04.11.gb_re_26_1/ocean/scaling/ablation/2026-04-11T16-37-30.751_ablation_ngpu6136"
]
ab = slurp_all(paths)
display(ab)


# ----------------------------------
# plot
# ----------------------------------
using Makie
using CairoMakie

colors = Makie.wong_colors()
co_color = colors[1]
ab_color = colors[2]

x_co  = co[:, 1]
y_co  = co[:, 2]
y_co_norm  = y_co ./ y_co[1]

x_ab  = ab[:, 1]
y_ab  = ab[:, 2]
y_ab_norm  = y_ab ./ y_co[1]

fig = Figure(size = (600, 400))
ax = Axis(
    fig[1, 1],
    # title = "Weak scaling",
    xlabel = "Number of GPUs",
    xscale = log2,
    # yscale = log2,
    xticks = unique!([x_co; x_ab]),
    ylabel = "Wallclock time [a.u.]",
)


# Lines for the datapoints
lines!(ax, x_co, y_co_norm; linewidth=2, color=co_color)
lines!(ax, x_ab, y_ab_norm; linewidth=2, color=ab_color)
# Scatter plot for the datapoints
scatter!(ax, x_co, y_co_norm; markersize = 10, color=co_color, marker=:circle, label="With communication optimizations")
scatter!(ax, x_ab, y_ab_norm; markersize = 10, color=ab_color, marker=:rect,   label="Without communication optimizations")

# Ideal scaling lines
hlines!(ax, y_co_norm[1], linestyle = :dash, color=co_color, linewidth=2)
hlines!(ax, y_ab_norm[1], linestyle = :dash, color=ab_color, linewidth=2)

# for logx
# Fake line not plotted, just for having an entry in the legend with a "neutral" color.
hlines!(ax, -1, linestyle = :dash, color=:grey, linewidth=2, label="Ideal weak scaling")
# Set y axis limits
ylims!(ax, 0, maximum([y_co_norm; y_ab_norm]) .* 1.1)

# # for loglog
# # Fake line not plotted, just for having an entry in the legend with a "neutral" color.
# hlines!(ax, 0.01, linestyle = :dash, color=:grey, linewidth=2, label="Ideal weak scaling")
# # Set y axis limits
# ylims!(ax, 0.5, maximum([y_co_norm; y_ab_norm]) .* 1.1)


# add labels slightly above each point
text!(
    ax,
    x_co,
    y_co_norm,
    text = string.(round.(y_co; digits=1)),
    align = (:center, :bottom),
    offset = (0, 10)  # shift upward a bit
)
for i in eachindex(x_ab)
    label = string(round(y_ab[i]; digits=1))
    offset = (0, 20)

    if x_ab[i] == 2048
        label = string(round(y_ab[i]; digits=1))
        offset = (0, 5)
    end

    text!(
        ax,
        [x_ab[i]],
        [y_ab_norm[i]],
        text = [label],
        align = (:center, :bottom),
        offset = offset,
    )
end

# Legend
axislegend(ax; position=:lt)

save("ablation.png", fig)
