module GBAnalyzer

using Dates, TranscodingStreams, CodecZlib, JSON, Statistics, PrettyTables

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
    only_or_default(list, default)

Return the single element of `list`, or `default` if `list` is empty.
"""
function only_or_default(list, default)
    length(list) == 0 ? default : only(list)
end

only_or_default(default)::Function = list -> only_or_default(list, default)

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
    get_jobid(pth::AbstractString)::Union{AbstractString, Invalid}

Finds the Slurm Job ID from the path `pth`
"""
function get_jobid(pth::AbstractString)::Union{AbstractString, Invalid}
    readdir(pth) |>
        Base.Fix1(filter, endswith(".err")) |>
        only_or_default(INVALID)            |>
        maybe(Base.Fix2(split, "."))        |>
        maybe(first)
end

"""
    profile_dir(
            root::AbstractString, jid::AbstractString, rank::Int;
            prof_dir::AbstractString="loop2"
        )::String

Returns the directory correspoding to scaling test at `root`, where the Slurm
Job ID is `jid`, and the MPI rank is `rank`.
"""
function profile_dir(
            root::AbstractString, jid::AbstractString, rank::Int;
            prof_dir::AbstractString="loop2"
        )::String
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
    ddir = readdir(pdir) |> only_or_default(INVALID)

    isinvalid(ddir) && return INVALID
    
    readdir(joinpath(pdir, ddir)) |>
        Base.Fix1(filter, endswith(".trace.json.gz")) |>
        only_or_default(INVALID)                      |>
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

# --------------------------------------------------------------------------

# --- Process / Manage Data from many Runs ---------------------------------

"""
    Run

Represents a single benchmark run.

# Fields
- `path::String`: Path to the run directory.
- `time::DateTime`: Timestamp of the run.
- `sim::String`: Simulation identifier.
- `ngpu::Vector{Int}`: GPU counts available for this run.
- `prof::Vector{Int}`: GPU counts that have profiling data.
- `ppath::Vector{String}`: Paths to the corresponding profile traces.
"""
struct Run
    path::String
    time::DateTime
    sim::String
    ngpu::Vector{Int}
    prof::Vector{Int}
    ppath::Vector{String}
end

"""
    summarize_run_dirs(prefix::String, pname::String)::Dict{String, Vector{Run}}

Generate a summary object of all the profiling data in a collection of scaling data.
This function traverses a directory tree, possibly containing multiple scaling runs
(eg. several repeated Slurm Jobs with different parameters, of different parts of the
same scaling run done in batches), and classifies each using a `Run` object.
"""
function summarize_run_dirs(prefix::String, pname::String)::Dict{String, Vector{Run}}
    runs = readdir(prefix) |> Base.Fix1(map, Base.Fix1(joinpath, prefix)) |> filter(isdir)
    runs_by_type = Dict{String, Vector{Run}}()
    for run in runs
        sim_dir = readdir(run)
        
        sim = sim_dir |> filter(endswith(".jl"))
        if length(sim) != 1
            continue
        end
        sim = only(sim)
        
        r = splitpath(run) |> last
        d = split(r, ".") |> first |> Base.Fix2(DateTime, dateformat"yyyy-mm-ddTHH-MM-SS")
        t = sim_dir  |> filter(startswith("ngpu=")) |>
            Base.Fix1(map, x->last(split(x, "="))) |> Base.Fix1(map, Base.Fix1(parse, Int))
        p = Int[]
        pp = String[]
        for i in t
            jid = get_jobid(joinpath(run, test_dir(i)))
            if isinvalid(jid)
                continue
            end
            
            pdir = profile_dir(joinpath(prefix, run), jid, 0; prof_dir=pname)
            if !ispath(pdir)
                continue
            end
            
            tpath = profile_for_job(pdir)
            if isinvalid(tpath)
                continue
            end
            push!(p, i)
            push!(pp, tpath)
        end
        
        push!(get!(runs_by_type, sim, []), Run(run, d, sim, t, p, pp))
    end
    runs_by_type
end

# --------------------------------------------------------------------------

# --- Display / Summarize Manage Data from many Runs -----------------------

"""
    get_ngpu_columns(data)

Create a list of GPU counts from a collection of runs -- uses `Set` to
eliminate duplicates, and `sort` to sort from smallest to largest
"""
get_ngpu_columns(data) = map(x->x.ngpu, data) |>
    Base.Fix1(reduce, vcat)                   |>
    Set                                       |>
    collect                                   |>
    sort

"""
    summary_table(data)

Create a `PrettyTable` representation of the run summary generated by
`summarize_run_dirs` -- valid runs (i.e. ones that contain profiling data)
are represented by a `1`, others are represented by a `0`
"""
function summary_table(data)
    ngpu = get_ngpu_columns(data)
    ngpu_h = get_ngpu_columns(data) |> Base.Fix1(map, string)
    header = vcat(["idx", "run"], ngpu_h)

    idx = collect(1:length(data))
    dates = map(x->x.time, data)
    paths = map(x->x.path, data)

    tdata = hcat(idx, paths)
    for n in ngpu
        z = zeros(Int, length(paths))
        for i = 1:length(z)
            if n in data[i].prof
                z[i] = 1
            end
        end
        tdata = hcat(tdata, z)
    end
    
    pretty_table(
        tdata;
        column_labels = header,
        table_format  = TextTableFormat(
            borders = text_table_borders__unicode_rounded
        ),
        fit_table_in_display_horizontally = false
    )
end

# --------------------------------------------------------------------------

end # module GBAnalyzer
