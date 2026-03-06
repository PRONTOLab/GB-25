using Reactant
using Oceananigans
import Oceananigans.TimeSteppers: time_step!, update_state!, maybe_initialize_state!
using Oceananigans.Architectures: ReactantState
using Reactant_jll: libReactantExtra

# No-op for Reactant: the iteration == 0 check evaluates at trace time,
# causing a redundant update_state! to be compiled into every time_step!.
# Instead, first_time_step! handles initialization explicitly.
maybe_initialize_state!(::Oceananigans.AbstractModel{<:Any, <:ReactantState}, callbacks) = nothing
maybe_initialize_state!(::Oceananigans.AbstractModel{<:Any, <:Oceananigans.Distributed{<:ReactantState}}, callbacks) = nothing

const TRY_COMPILE_FAILED = Ref(false)

function try_compile_code(f)
    try
        f()
    catch e
        @error "Failed to compile" exception=(e, catch_backtrace())
        TRY_COMPILE_FAILED[] = true
        Text("""
        // Failed to compile
        //$e
        """)
    end
end

function first_time_step!(model)
    Reactant.Profiler.annotate("first_time_step") do
        Oceananigans.initialize!(model)
        Oceananigans.TimeSteppers.update_state!(model)
        Δt = model.clock.last_Δt + 0
        Oceananigans.TimeSteppers.time_step!(model, Δt; euler=true)
    end
    return nothing
end

function time_step!(model)
    Reactant.Profiler.annotate("time_step") do
        Δt = model.clock.last_Δt + 0
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

function loop!(model, Ninner)
    Reactant.Profiler.annotate("loop") do
        Δt = model.clock.last_Δt + 0
        @trace track_numbers=false for _ = 1:Ninner
            Oceananigans.TimeSteppers.time_step!(model, Δt)
        end
    end
    return nothing
end

function preamble(; rendezvous_warn::Union{Nothing,Int}=nothing, rendezvous_terminate::Union{Nothing,Int}=nothing)
    # If we are in GitHub Actions, make `TMPDIR` be a local directory from which we
    # can upload artifacts at the end.
    if get(ENV, "GITHUB_ACTIONS", "false") == "true"
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
        ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "..", "tmp"))
    end

    # Unset environment variables which would cause XLA distributed to hang indefinitely.
    for key in ("no_proxy", "http_proxy", "https_proxy", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY")
        delete!(ENV, key)
    end

    if rendezvous_warn isa Int || rendezvous_terminate isa Int
        error("""
              Setting rendezvous timeouts in `preamble` is not supported anymore.
              Use `XLA_FLAGS` instead, e.g.
                  XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80"
              """)
    end
end
