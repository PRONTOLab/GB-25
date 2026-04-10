using ArgParse
using GordonBell25
using Oceananigans
using Reactant

const STAGE_DEPS = Dict(
    :initialize => Symbol[],
    :update_state => [:initialize],
    :first_time_step => [:initialize, :update_state],
    :time_step => [:initialize, :update_state, :first_time_step],
    :loop => [:initialize, :update_state, :first_time_step],
)

const DEFAULT_RELOAD_FILES = [
    "src/timestepping_utils.jl",
    "src/correctness.jl",
    "src/baroclinic_instability_model.jl",
]

mutable struct DebugHarness
    parsed_args::Dict{String,Any}
    package_root::String
    default_float_type
    compile_options
    rtol::Float64
    atol::Float64
    compiled::Dict{Any,Any}
end

function parse_cli_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--grid-x"
            help = "Total x points including halos."
            default = 32
            arg_type = Int
        "--grid-y"
            help = "Total y points including halos."
            default = 32
            arg_type = Int
        "--grid-z"
            help = "Total z points."
            default = 8
            arg_type = Int
        "--float-type"
            help = "Model float type."
            default = "Float32"
            arg_type = String
        "--target-float-type"
            help = "Execution float type or empty string for no lowering."
            default = "BFloat16"
            arg_type = String
        "--limbs"
            help = "Number of lower-precision limbs in the multifloat lowering."
            default = 2
            arg_type = Int
        "--dimension"
            help = "Multifloat expansion dimension (first, last, tuple)."
            default = "first"
            arg_type = String
        "--loop-count"
            help = "Default loop trip count for loop-specific helpers."
            default = 100
            arg_type = Int
    end
    return ArgParse.parse_args(ARGS, s)
end

function make_compile_options(parsed_args)
    return CompileOptions(;
        sync = true,
        raise = true,
        strip_llvm_debuginfo = true,
        strip = :all,
        multifloat = GordonBell25.multifloat_from_args(parsed_args),
    )
end

function init_harness(parsed_args)
    # Keep tolerance semantics identical to correctness_sharded_baroclinic_instability_simulation_run.jl.
    default_float_type = GordonBell25.float_type_from_args(parsed_args)
    Oceananigans.defaults.FloatType = default_float_type

    Reactant.set_default_backend("cpu")
    GordonBell25.initialize(; single_gpu_per_process = false)

    Ndev = length(Reactant.devices())
    @info "Device count" Ndev
    Ndev == 1 || error("This harness currently expects a single device; got Ndev = $Ndev")

    package_root = dirname(dirname(pathof(GordonBell25)))
    return DebugHarness(
        parsed_args,
        package_root,
        default_float_type,
        make_compile_options(parsed_args),
        sqrt(eps(default_float_type)),
        0.0,
        Dict{Any,Any}(),
    )
end

function grid_shape(h::DebugHarness)
    H = 8
    Nx = h.parsed_args["grid-x"] - 2H
    Ny = h.parsed_args["grid-y"] - 2H
    Nz = h.parsed_args["grid-z"]
    Nx > 0 || error("--grid-x must be > $(2H)")
    Ny > 0 || error("--grid-y must be > $(2H)")
    return (; H, Nx, Ny, Nz)
end

function build_model_pair(h::DebugHarness)
    (; H, Nx, Ny, Nz) = grid_shape(h)

    model_kw = (
        halo = (H, H, H),
        Δt = 1e-9,
    )

    rmodel = GordonBell25.baroclinic_instability_model(
        Oceananigans.ReactantState(),
        Nx,
        Ny,
        Nz;
        model_kw...,
    )
    vmodel = GordonBell25.baroclinic_instability_model(CPU(), Nx, Ny, Nz; model_kw...)

    ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
    vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
    set!(vmodel, u = ui, v = vi)
    GordonBell25.sync_states!(rmodel, vmodel)

    return (; rmodel, vmodel)
end

function build_reactant_pair(h::DebugHarness)
    pair = build_model_pair(h)
    (; H, Nx, Ny, Nz) = grid_shape(h)

    model_kw = (
        halo = (H, H, H),
        Δt = 1e-9,
    )

    rmodel2 = GordonBell25.baroclinic_instability_model(
        Oceananigans.ReactantState(),
        Nx,
        Ny,
        Nz;
        model_kw...,
    )
    GordonBell25.sync_states!(rmodel2, pair.vmodel)

    return (; rmodel_a = pair.rmodel, rmodel_b = rmodel2, vmodel = pair.vmodel)
end

function compare_models!(h::DebugHarness, model_a, model_b; label::String="", throw_error::Bool=true)
    isempty(label) || @info label
    try
        GordonBell25.compare_states(
            model_a,
            model_b;
            include_halos = true,
            throw_error = true,
            rtol = h.rtol,
            atol = h.atol,
        )
        return true
    catch err
        throw_error && rethrow()
        return false
    end
end

function stage_key(stage::Symbol; loop_count::Union{Nothing,Int}=nothing)
    return stage == :loop ? (stage, something(loop_count)) : stage
end

function run_compiled_stage!(h::DebugHarness, stage::Symbol, rmodel, vmodel; count::Int=1)
    if stage == :initialize
        h.compiled[:initialize](rmodel)
        Oceananigans.initialize!(vmodel)
    elseif stage == :update_state
        h.compiled[:update_state](rmodel)
        Oceananigans.TimeSteppers.update_state!(vmodel)
    elseif stage == :first_time_step
        h.compiled[:first_time_step](rmodel)
        GordonBell25.first_time_step!(vmodel)
    elseif stage == :time_step
        for _ in 1:count
            h.compiled[:time_step](rmodel)
            GordonBell25.time_step!(vmodel)
        end
    elseif stage == :loop
        loop_key = stage_key(:loop; loop_count = count)
        h.compiled[loop_key](rmodel, ConcreteRNumber(count))
        GordonBell25.loop!(vmodel, count)
    else
        error("Unknown stage $stage")
    end
    return nothing
end

function run_stage_on_reactant!(h::DebugHarness, stage::Symbol, rmodel; count::Int=1)
    if stage == :initialize
        h.compiled[:initialize](rmodel)
    elseif stage == :update_state
        h.compiled[:update_state](rmodel)
    elseif stage == :first_time_step
        h.compiled[:first_time_step](rmodel)
    elseif stage == :time_step
        for _ in 1:count
            h.compiled[:time_step](rmodel)
        end
    elseif stage == :loop
        loop_key = stage_key(:loop; loop_count = count)
        h.compiled[loop_key](rmodel, ConcreteRNumber(count))
    else
        error("Unknown stage $stage")
    end
    return nothing
end

function prepare_state_for_compile!(h::DebugHarness, stage::Symbol, scratch; loop_count::Int)
    for dep in STAGE_DEPS[stage]
        @info "Preparing prerequisite for compile" stage dep
        ensure_compiled!(h, dep; loop_count)
        run_compiled_stage!(h, dep, scratch.rmodel, scratch.vmodel; count = loop_count)
    end
    return nothing
end

function ensure_compiled!(h::DebugHarness, stage::Symbol; loop_count::Int=h.parsed_args["loop-count"])
    key = stage_key(stage; loop_count)
    haskey(h.compiled, key) && return h.compiled[key]

    scratch = build_model_pair(h)
    prepare_state_for_compile!(h, stage, scratch; loop_count)

    @info "Compiling stage thunk" stage key
    thunk = if stage == :initialize
        @compile compile_options = h.compile_options Oceananigans.initialize!(scratch.rmodel)
    elseif stage == :update_state
        @compile compile_options = h.compile_options Oceananigans.TimeSteppers.update_state!(scratch.rmodel)
    elseif stage == :first_time_step
        @compile compile_options = h.compile_options GordonBell25.first_time_step!(scratch.rmodel)
    elseif stage == :time_step
        @compile compile_options = h.compile_options GordonBell25.time_step!(scratch.rmodel)
    elseif stage == :loop
        rcount = ConcreteRNumber(loop_count)
        @compile compile_options = h.compile_options GordonBell25.loop!(scratch.rmodel, rcount)
    else
        error("Unknown stage $stage")
    end

    h.compiled[key] = thunk
    @info "Compiled stage" stage key
    return thunk
end

function compile_common!(h::DebugHarness; loop_count::Int=h.parsed_args["loop-count"])
    ensure_compiled!(h, :initialize; loop_count)
    ensure_compiled!(h, :update_state; loop_count)
    ensure_compiled!(h, :first_time_step; loop_count)
    ensure_compiled!(h, :time_step; loop_count)
    ensure_compiled!(h, :loop; loop_count)
    return nothing
end

function clear_compiled!(h::DebugHarness)
    empty!(h.compiled)
    @info "Cleared compiled thunk cache"
    return nothing
end

function reload!(h::DebugHarness, relpaths::Vector{String}=DEFAULT_RELOAD_FILES)
    for relpath in relpaths
        abspath = isabspath(relpath) ? relpath : joinpath(h.package_root, relpath)
        @info "Reloading file" abspath
        Base.include(GordonBell25, abspath)
    end
    clear_compiled!(h)
    return nothing
end

function check_stage!(
    h::DebugHarness,
    stage::Symbol;
    time_step_count::Int=1,
    loop_count::Int=h.parsed_args["loop-count"],
    compare_initial::Bool=true,
    compare_after_prereqs::Bool=true,
    throw_error::Bool=true,
)
    pair = build_model_pair(h)
    compare_initial && compare_models!(h, pair.rmodel, pair.vmodel; label = "Initial comparison:", throw_error)

    if stage == :loop
        ensure_compiled!(h, :loop; loop_count)
    else
        ensure_compiled!(h, stage; loop_count)
    end

    for dep in STAGE_DEPS[stage]
        @info "Running prerequisite stage" stage dep
        run_compiled_stage!(h, dep, pair.rmodel, pair.vmodel; count = loop_count)
    end

    if compare_after_prereqs && !isempty(STAGE_DEPS[stage])
        compare_models!(
            h,
            pair.rmodel,
            pair.vmodel;
            label = "After prerequisites for $(stage):",
            throw_error,
        )
    end

    if stage == :time_step
        run_compiled_stage!(h, :time_step, pair.rmodel, pair.vmodel; count = time_step_count)
        return compare_models!(
            h,
            pair.rmodel,
            pair.vmodel;
            label = "After time_step! x$(time_step_count):",
            throw_error,
        )
    elseif stage == :loop
        run_compiled_stage!(h, :loop, pair.rmodel, pair.vmodel; count = loop_count)
        return compare_models!(
            h,
            pair.rmodel,
            pair.vmodel;
            label = "After loop!($(loop_count)):",
            throw_error,
        )
    else
        run_compiled_stage!(h, stage, pair.rmodel, pair.vmodel)
        return compare_models!(h, pair.rmodel, pair.vmodel; label = "After $(stage):", throw_error)
    end
end

function check_loop_vs_steps!(
    h::DebugHarness;
    loop_count::Int=h.parsed_args["loop-count"],
    throw_error::Bool=true,
)
    pair = build_reactant_pair(h)

    ensure_compiled!(h, :loop; loop_count)
    ensure_compiled!(h, :time_step; loop_count)

    for dep in STAGE_DEPS[:loop]
        run_stage_on_reactant!(h, dep, pair.rmodel_a; count = loop_count)
        run_stage_on_reactant!(h, dep, pair.rmodel_b; count = loop_count)
    end

    run_stage_on_reactant!(h, :loop, pair.rmodel_a; count = loop_count)
    run_stage_on_reactant!(h, :time_step, pair.rmodel_b; count = loop_count)

    return compare_models!(
        h,
        pair.rmodel_a,
        pair.rmodel_b;
        label = "Loop thunk vs repeated time_step thunk after $(loop_count) steps:",
        throw_error,
    )
end

function find_first_bad_timestep!(
    h::DebugHarness;
    max_steps::Int=h.parsed_args["loop-count"],
)
    ok = check_stage!(h, :time_step; time_step_count = max_steps, compare_initial = false, throw_error = false)
    ok && return println("No discrepancy was found through $(max_steps) repeated time_step! calls.")

    lo = 0
    hi = max_steps
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        ok = check_stage!(h, :time_step; time_step_count = mid, compare_initial = false, throw_error = false)
        if ok
            lo = mid
        else
            hi = mid
        end
    end

    println("First failing repeated time_step! count appears to be $(hi).")
    return hi
end

function status(h::DebugHarness)
    grid_x = h.parsed_args["grid-x"]
    grid_y = h.parsed_args["grid-y"]
    grid_z = h.parsed_args["grid-z"]
    float_type = h.parsed_args["float-type"]
    target_float_type = h.parsed_args["target-float-type"]
    loop_count = h.parsed_args["loop-count"]
    println("grid = ($(grid_x), $(grid_y), $(grid_z))")
    println("float-type = $(float_type), target-float-type = $(target_float_type)")
    println("default loop-count = $(loop_count)")
    println("compiled keys = $(collect(keys(h.compiled)))")
    return nothing
end

function print_help()
    println(
        """
Available helpers:
  status(dbg)
  compile_common!(dbg; loop_count=100)
  clear_compiled!(dbg)
  reload!(dbg)  # re-include selected GordonBell25 source files and clear thunks

Correctness checks:
  check_stage!(dbg, :initialize)
  check_stage!(dbg, :update_state)
  check_stage!(dbg, :first_time_step)
  check_stage!(dbg, :time_step; time_step_count=1)
  check_stage!(dbg, :loop; loop_count=100)

Isolation helpers:
  check_loop_vs_steps!(dbg; loop_count=100)
    Compare compiled loop! against compiled repeated time_step! on Reactant-only models.
  find_first_bad_timestep!(dbg; max_steps=100)
    Binary-search the first failing repeated time_step! count.

Typical edit/run cycle without Revise:
  reload!(dbg, [\"src/timestepping_utils.jl\"])
  check_stage!(dbg, :loop; loop_count=100)
        """,
    )
    return nothing
end

# Persistent correctness harness quick notes:
#
# Exact command to run from the root of the GB-25 repo:
#   julia --project=. -O0 --threads=16 -i analysis/persistent_correctness_harness.jl \
#     --grid-x 32 --grid-y 32 --grid-z 8 \
#     --float-type Float32 --target-float-type BFloat16 \
#     --dimension first --loop-count 100
#
# The useful commands in that REPL are:
#   status(dbg)
#   compile_common!(dbg; loop_count=100)
#
#   check_stage!(dbg, :initialize)
#   check_stage!(dbg, :update_state)
#   check_stage!(dbg, :first_time_step)
#   check_stage!(dbg, :time_step; time_step_count=1)
#   check_stage!(dbg, :loop; loop_count=100)
#
# For isolating where the bug actually lives:
#   check_loop_vs_steps!(dbg; loop_count=100)
#   find_first_bad_timestep!(dbg; max_steps=100)
#
# Those are the important ones:
# - `check_stage!(dbg, :loop; loop_count=100)` tests the full `loop!` correctness against CPU.
# - `check_loop_vs_steps!(dbg; loop_count=100)` compares compiled `loop!` against compiled repeated
#   `time_step!` on Reactant-only models. If this fails, the bug is in `loop!` specifically, not in
#   `time_step!`.
# - `find_first_bad_timestep!(dbg; max_steps=100)` binary-searches the first failing repeated
#   `time_step!` count, which is much better than linearly stepping through 100 iterations.
#
# Since Revise is off the table, use manual reload support:
#   reload!(dbg, ["src/timestepping_utils.jl"])
#   reload!(dbg, ["src/timestepping_utils.jl", "src/baroclinic_instability_model.jl"])
#
# That re-includes the selected source files into `GordonBell25` and clears the compiled thunk cache,
# so the typical edit/run cycle becomes:
#   reload!(dbg, ["src/timestepping_utils.jl"])
#   check_stage!(dbg, :loop; loop_count=100)
const dbg = init_harness(parse_cli_args())

print_help()
println("\nHarness ready as `dbg`.")
