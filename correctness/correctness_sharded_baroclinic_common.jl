using GordonBell25
using Oceananigans
using Random
using Reactant

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 8,
    grid_y_default = 4,
    grid_z_default = 4,
)

const default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type

Reactant.set_default_backend("tpu")

if !GordonBell25.is_distributed_env_present()
    using MPI
    MPI.Init()
end

const THROW_ERROR = true
const INCLUDE_HALOS = true
const RTOL = sqrt(eps(default_float_type))
const ATOL = 0
const DEFAULT_SEED = 42

function timed_phase(f::Function, label)
    @info "$label [start]"
    t0 = time()
    result = f()
    @info "$label [done]" elapsed_s = round(time() - t0; digits=3)
    return result
end

timed_phase(label, f::Function) = timed_phase(f, label)

function setup_context(; seed = DEFAULT_SEED)
    Ndev = length(Reactant.devices())
    skip_distributed_init = get(ENV, "GB_SKIP_DISTRIBUTED_INIT", "0") == "1"
    if Ndev > 1 && !skip_distributed_init && GordonBell25.is_distributed_env_present()
        GordonBell25.initialize(; single_gpu_per_process=false)
        Ndev = length(Reactant.devices())
    elseif Ndev > 1 && (skip_distributed_init || !GordonBell25.is_distributed_env_present())
        @info "Skipping GordonBell25.initialize for local single-process multi-device run"
    end
    @show Ndev

    Rx, Ry = Ndev == 1 ? (1, 1) : GordonBell25.factors(Ndev)

    rarch = Oceananigans.ReactantState()
    if Ndev != 1
        rarch = Oceananigans.Distributed(
            rarch;
            partition = Partition(Rx, Ry, 1),
        )
    end

    H = Ndev == 1 ? 4 : 8
    Nz = parsed_args["grid-z"]
    if Ndev == 1
        Nx = parsed_args["grid-x"]
        Ny = parsed_args["grid-y"]
    else
        Tx = parsed_args["grid-x"] * Rx
        Ty = parsed_args["grid-y"] * Ry
        Nx = Tx - 2H
        Ny = Ty - 2H
    end

    Nx > 0 || throw(ArgumentError("Nx must be positive; got Nx = $Nx"))
    Ny > 0 || throw(ArgumentError("Ny must be positive; got Ny = $Ny"))

    model_kw = (
        halo = (H, H, H),
        Δt = 1e-9,
    )

    varch = CPU()
    rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
    vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
    @show vmodel
    @show rmodel

    Random.seed!(seed)
    ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
    vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
    set!(vmodel, u=ui, v=vi)
    GordonBell25.sync_states!(rmodel, vmodel)

    compile_options = CompileOptions(;
        sync=true,
        raise=true,
        strip_llvm_debuginfo=true,
        strip=:all,
        multifloat=GordonBell25.multifloat_from_args(parsed_args),
    )

    return (;
        rmodel,
        vmodel,
        compile_options,
        include_halos=INCLUDE_HALOS,
        throw_error=THROW_ERROR,
        rtol=RTOL,
        atol=ATOL,
    )
end

function compare!(ctx, label)
    @info label
    GordonBell25.compare_states(
        ctx.rmodel,
        ctx.vmodel;
        include_halos=ctx.include_halos,
        throw_error=ctx.throw_error,
        rtol=ctx.rtol,
        atol=ctx.atol,
    )
    return nothing
end

function initialize_and_update_state!(ctx)
    Oceananigans.initialize!(ctx.vmodel)
    Oceananigans.TimeSteppers.update_state!(ctx.vmodel)

    timed_phase("Reactant initialize!(rmodel)") do
        @jit compile_options=ctx.compile_options Oceananigans.initialize!(ctx.rmodel)
    end
    timed_phase("Reactant update_state!(rmodel)") do
        @jit compile_options=ctx.compile_options Oceananigans.TimeSteppers.update_state!(ctx.rmodel)
    end
    return nothing
end

function run_first_time_step!(ctx)
    GordonBell25.sync_states!(ctx.rmodel, ctx.vmodel)
    @showtime GordonBell25.first_time_step!(ctx.vmodel)

    rfirst! = timed_phase("Compile first_time_step!(rmodel)") do
        @compile compile_options=ctx.compile_options GordonBell25.first_time_step!(ctx.rmodel)
    end
    @showtime rfirst!(ctx.rmodel)
    return nothing
end

function compile_time_step!(ctx)
    return timed_phase("Compile time_step!(rmodel)") do
        @compile compile_options=ctx.compile_options GordonBell25.time_step!(ctx.rmodel)
    end
end

function resync_and_update_state!(ctx)
    GordonBell25.sync_states!(ctx.rmodel, ctx.vmodel)
    @jit compile_options=ctx.compile_options Oceananigans.TimeSteppers.update_state!(ctx.rmodel)
    return nothing
end
