using ArgParse
using GordonBell25
using Oceananigans
using Reactant
using Reactant.Profiler

const STAGES = ("initialize", "update_state", "first_time_step", "time_step", "loop")
const STAGE_DEPENDENCIES = Dict(
    "initialize" => String[],
    "update_state" => ["initialize"],
    "first_time_step" => ["initialize", "update_state"],
    "time_step" => ["initialize", "update_state", "first_time_step"],
    "loop" => ["initialize", "update_state", "first_time_step"],
)

function parse_cli_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--mode"
            help = "Either `correctness` or `profile`."
            default = "correctness"
            arg_type = String
        "--ops"
            help = "Comma-separated stages to test in correctness mode."
            default = "time_step"
            arg_type = String
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
            help = "Model float type (Float64/f64, Float32/f32, Float16/f16, BFloat16/bf16)."
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
        "--nrepeat"
            help = "Number of profiled rstep invocations in profile mode."
            default = 1
            arg_type = Int
        "--warmup"
            help = "Number of warmup invocations before profiling in profile mode."
            default = 1
            arg_type = Int
        "--time-step-count"
            help = "Number of `time_step!` calls in correctness mode."
            default = 1
            arg_type = Int
        "--loop-count"
            help = "Loop trip count for `loop!` in correctness mode."
            default = 2
            arg_type = Int
        "--top"
            help = "Number of framework ops to print in profile mode."
            default = 15
            arg_type = Int
        "--profile-dir"
            help = "Directory for profiler outputs. Empty means use a temporary scratch dir."
            default = ""
            arg_type = String
    end
    return ArgParse.parse_args(ARGS, s)
end

function normalize_mode(mode::String)
    mode = lowercase(strip(mode))
    mode in ("correctness", "profile") || error("Unknown --mode=$mode")
    return mode
end

function parse_requested_ops(ops_string::String)
    ops = String[]
    for op in split(lowercase(ops_string), ',')
        op = strip(op)
        isempty(op) && continue
        op in STAGES || error("Unknown stage `$op`. Expected one of $(join(STAGES, ", ")).")
        op in ops || push!(ops, op)
    end
    isempty(ops) && error("No stages were requested.")
    return ops
end

function default_rtol(float_type)
    return sqrt(eps(float_type))
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

function build_models(parsed_args)
    default_float_type = GordonBell25.float_type_from_string(parsed_args["float-type"])
    Oceananigans.defaults.FloatType = default_float_type

    Reactant.set_default_backend("cpu")
    GordonBell25.initialize(; single_gpu_per_process = false)

    Ndev = length(Reactant.devices())
    @info "Device count" Ndev
    Ndev == 1 || error("This script currently expects a single device; got Ndev = $Ndev")

    H = 8
    Nx = parsed_args["grid-x"] - 2H
    Ny = parsed_args["grid-y"] - 2H
    Nz = parsed_args["grid-z"]
    Nx > 0 || error("--grid-x must be > $(2H)")
    Ny > 0 || error("--grid-y must be > $(2H)")

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

    return (;
        default_float_type,
        rmodel,
        vmodel,
        compile_options = make_compile_options(parsed_args),
    )
end

function compare_models!(state; label::String)
    @info label
    GordonBell25.compare_states(
        state.rmodel,
        state.vmodel;
        include_halos = true,
        throw_error = true,
        rtol = default_rtol(state.default_float_type),
        atol = 0,
    )
end

function print_top_framework_ops(xplane_file::String, top::Int)
    reports = Reactant.Profiler.get_framework_op_stats(xplane_file)
    isempty(reports) && return println("No framework op stats were available.")

    sorted_reports = sort(reports; by = r -> r.total_self_time_in_us, rev = true)
    println("\nTop framework ops by self time:")
    Reactant.Profiler.print_framework_op_stats(sorted_reports[1:min(top, length(sorted_reports))])
end

function run_stage!(stage::String, state, parsed_args, compiled)
    rmodel = state.rmodel
    vmodel = state.vmodel
    compile_options = state.compile_options

    if stage == "initialize"
        @info "Stage initialize [start]"
        @jit compile_options = compile_options Oceananigans.initialize!(rmodel)
        Oceananigans.initialize!(vmodel)
    elseif stage == "update_state"
        @info "Stage update_state [start]"
        @jit compile_options = compile_options Oceananigans.TimeSteppers.update_state!(rmodel)
        Oceananigans.TimeSteppers.update_state!(vmodel)
    elseif stage == "first_time_step"
        @info "Stage first_time_step [start]"
        rfirst! = get!(compiled, :first_time_step) do
            @compile compile_options = compile_options GordonBell25.first_time_step!(rmodel)
        end
        rfirst!(rmodel)
        GordonBell25.first_time_step!(vmodel)
    elseif stage == "time_step"
        count = parsed_args["time-step-count"]
        @info "Stage time_step [start]" count
        rstep! = get!(compiled, :time_step) do
            @compile compile_options = compile_options GordonBell25.time_step!(rmodel)
        end
        for _ in 1:count
            rstep!(rmodel)
            GordonBell25.time_step!(vmodel)
        end
    elseif stage == "loop"
        count = parsed_args["loop-count"]
        @info "Stage loop [start]" count
        rcount = ConcreteRNumber(count)
        rloop! = get!(compiled, :loop) do
            @compile compile_options = compile_options GordonBell25.loop!(rmodel, rcount)
        end
        rloop!(rmodel, rcount)
        GordonBell25.loop!(vmodel, count)
    else
        error("Unknown stage `$stage`.")
    end

    compare_models!(state; label = "After $stage:")
end

function run_correctness_mode(parsed_args)
    requested_ops = parse_requested_ops(parsed_args["ops"])

    for stage in requested_ops
        println("\n=== Testing $stage ===")
        state = build_models(parsed_args)
        compare_models!(state; label = "At the beginning:")

        compiled = Dict{Symbol,Any}()
        executed = Set{String}()

        function ensure_stage!(stage_name::String)
            for dep in STAGE_DEPENDENCIES[stage_name]
                dep in executed || ensure_stage!(dep)
            end
            if !(stage_name in executed)
                run_stage!(stage_name, state, parsed_args, compiled)
                push!(executed, stage_name)
            end
            return nothing
        end

        ensure_stage!(stage)
    end
end

function run_profile_mode(parsed_args)
    state = build_models(parsed_args)

    @info "Priming initialize! and update_state! before profiling rstep!"
    @jit compile_options = state.compile_options Oceananigans.initialize!(state.rmodel)
    Oceananigans.initialize!(state.vmodel)

    @jit compile_options = state.compile_options Oceananigans.TimeSteppers.update_state!(state.rmodel)
    Oceananigans.TimeSteppers.update_state!(state.vmodel)

    GordonBell25.sync_states!(state.rmodel, state.vmodel)

    profile_dir = isempty(parsed_args["profile-dir"]) ? nothing : parsed_args["profile-dir"]
    profiled = Reactant.Profiler.profile_with_xprof(
        GordonBell25.time_step!,
        state.rmodel;
        nrepeat = parsed_args["nrepeat"],
        warmup = parsed_args["warmup"],
        profile_dir = profile_dir,
        compile_options = state.compile_options,
    )

    println("\nAggregate profile:")
    println(profiled.profiling_result)
    println("\nTrace file:")
    println(profiled.xplane_file)

    print_top_framework_ops(profiled.xplane_file, parsed_args["top"])
end

function main()
    parsed_args = parse_cli_args()
    mode = normalize_mode(parsed_args["mode"])

    if mode == "correctness"
        run_correctness_mode(parsed_args)
    else
        run_profile_mode(parsed_args)
    end
end

main()
