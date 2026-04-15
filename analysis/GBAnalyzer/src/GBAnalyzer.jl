module GBAnalyzer

# --- Monadic Error Handling -----------------------------------------------

struct Invalid end
const INVALID = Invalid()
isinvalid(x) = x isa Invalid

"""
    maybe(f)

Wrap `f` so that it propagates `INVALID` values untouched.
"""
maybe(f) = x -> isinvalid(x) ? INVALID : f(x)

"""
    only_or_default(list, default=INVALID)

Return the single element of `list`, or `default` if `list` is empty.
"""
function only_or_default(list, default=INVALID)
    length(list) == 0 ? default : only(list)
end

only_or_default(default) = list -> only_or_default(list, default)

# --------------------------------------------------------------------------

# --- Find Profiler Output -------------------------------------------------
"""
    test_dir(ng::Int)::String

Returns the name of the specific test directory correspoidng to a run with `ng` gpus
"""
function test_dir(ng::Int)::String
    nstr = lpad(ng, 5, "0")
    "ngpu=$(nstr)"
end

"""
    get_jobid(pth::String)::Union{String, Invalid}

Finds the Slurm Job ID from the path `pth`
"""
function get_jobid(pth::String)::Union{String, Invalid}
    readdir(pth) |>
        Base.Fix1(filter, endswith(".err")) |>
        only_or_default()                   |>
        maybe(Base.Fix2(split, "."))        |>
        maybe(first)
end

"""
    profile_dir(root::String, jid::String, rank::Int; prof_dir::String="loop2")::String

Returns the directory correspoding to scaling test at `root`, where the Slurm
Job ID is `jid`, and the MPI rank is `rank`.
"""
function profile_dir(root::String, jid::String, rank::Int; prof_dir::String="loop2")::String
    return joinpath(
        root, "profiling", "$(jid).$(rank)", prof_dir, "plugins", "profile"
    )
end

"""
    profile_for_job(pdir::String)::Union{String, Invalid}

Returns profiling trace file at location `pdir` -- will return `Invalid` if
that location does not contain a `.trave.json.gz`.
"""
function profile_for_job(pdir::String)::Union{String, Invalid}
    ddir = readdir(pdir) |> only_or_default()

    isinvalid(ddir) && return INVALID
    
    readdir(joinpath(pdir, ddir)) |>
        Base.Fix1(filter, endswith(".trace.json.gz")) |>
        only_or_default()                             |>
        maybe(name -> joinpath(pdir, ddir, name))
end

# --------------------------------------------------------------------------

# --- Read Profiler Output -------------------------------------------------
"""
    get_nccl_stats(pth::String)::Tuple{Dict, Dict}

Loads the profiler traced at `pth` and returns a dictionary containing the
NCCL events.

# Returns
- A Dict containing mean, std, sum, and frequency
- A Dict of all NCCL events: event name => event
"""
function get_nccl_stats(pth::String)::Tuple{Dict, Dict}
    stream = GzipDecompressorStream(open(pth))
    tprof  = readchomp(stream) |> JSON.parse
    close(stream)

    nccl_events = tprof["traceEvents"] |> 
        filter(x->"name" in keys(x) && contains(x["name"], "nccl"))

    nccl_stats = Dict()
    for event in nccl_events
        if !("name" in keys(event)) || !("dur" in keys(event))
            continue
        end
        push!(get!(nccl_stats, event["name"], []), event["dur"])
    end

    nccl_summary = Dict()
    for event_type in keys(nccl_stats)
        events = nccl_stats[event_type]
        nccl_summary[event_type] = (
            mean = mean(events),
            std  = std(events),
            sum  = sum(events),
            freq = length(events)
        )
    end

    nccl_summary, nccl_stats
end

"""
    get_run_stats(pth::String)::Tuple{Dict, Dict}

Loads the profiler traced at `pth` and returns a dictionary containing the
profiler events.

# Returns
- A Dict containing mean, std, sum, and frequency
- A Dict of all events: event name => event
"""
function get_run_stats(pth::String)::Tuple{Dict, Dict}
    stream = GzipDecompressorStream(open(pth))
    tprof  = readchomp(stream) |> JSON.parse
    close(stream)

    run_stats = Dict()
    for event in tprof["traceEvents"]
        if !("name" in keys(event)) || !("dur" in keys(event))
            continue
        end
        push!(get!(run_stats, event["name"], []), event["dur"])
    end

    run_summary = Dict()
    for event_type in keys(run_stats)
        events = run_stats[event_type]
        run_summary[event_type] = (
            mean = mean(events),
            std  = std(events),
            sum  = sum(events),
            freq = length(events)
        )
    end

    run_summary, run_stats
end


end # module GBAnalyzer
