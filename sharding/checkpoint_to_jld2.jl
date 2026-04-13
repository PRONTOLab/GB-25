#=
Convert serialized checkpoint directories to JLD2 for visualization.

Usage:
    julia --project=.. -O0 checkpoint_to_jld2.jl <checkpoint_dir> [<output.jld2>]

Examples:
    # Single checkpoint directory (auto-names output)
    julia --project=.. -O0 checkpoint_to_jld2.jl runs/2026-04-10T.../output/1823496.0/block_0001/output

    # Explicit output path
    julia --project=.. -O0 checkpoint_to_jld2.jl runs/.../block_0001/output  block1.jld2

    # Batch: convert all block_* dirs under a run
    for d in runs/.../output/*/block_*/output; do
        julia --project=.. -O0 checkpoint_to_jld2.jl "$d"
    done
=#

using GordonBell25
using JLD2

function main()
    if isempty(ARGS)
        error("""
            Usage: julia --project=.. -O0 checkpoint_to_jld2.jl <checkpoint_dir> [<output.jld2>]

            checkpoint_dir  Directory containing fields_rank*.dat files
            output.jld2     Optional output path (default: <checkpoint_dir>.jld2)
        """)
    end

    checkpoint_dir = ARGS[1]
    isdir(checkpoint_dir) || error("Not a directory: $checkpoint_dir")

    outfile = if length(ARGS) >= 2
        ARGS[2]
    else
        rstrip(checkpoint_dir, '/') * ".jld2"
    end

    @info "Loading checkpoint" checkpoint_dir
    meta, fields = GordonBell25.load_all_fields(checkpoint_dir)

    @info "Fields loaded" names=collect(keys(fields)) sizes=[size(v) for v in values(fields)]

    jldopen(outfile, "w") do file
        file["iteration"] = meta.iteration
        file["time"] = meta.time
        file["field_names"] = meta.field_names
        if meta.slices !== nothing
            file["slices"] = meta.slices
        end
        if meta.halo_sizes !== nothing
            file["halo_sizes"] = collect(meta.halo_sizes)
        end
        for (name, arr) in fields
            file[string(name)] = arr
        end
    end

    @info "Wrote $(filesize(outfile)) bytes" outfile
end

main()
