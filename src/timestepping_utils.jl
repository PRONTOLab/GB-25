using Reactant
using Oceananigans
import Oceananigans.TimeSteppers: first_time_step!, time_step!
using Reactant_jll: libReactantExtra

function try_code_hlo(f)
    try
        f()
    catch e
        @error "Failed to compile" exception=(e, catch_backtrace())
        global failed = true
        Text("""
        // Failed to compile
        //$e
        """)
    end
end

function first_time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

function time_step!(model)
    Δt = model.clock.last_Δt + 0
    Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end

function loop!(model, Ninner)
    Δt = model.clock.last_Δt + 0
    @trace track_numbers=false for _ = 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

function preamble(; rendezvous_warn::Union{Nothing,Int}=nothing, rendezvous_terminate::Union{Nothing,Int}=nothing)
    # If we are in GitHub Actions, make `TMPDIR` be a local directory from which we
    # can upload artifacts at the end.
    if get(ENV, "GITHUB_ACTIONS", "false") == "true"
        ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "..", "tmp"))
    end

    # Unset environment variables which would cause XLA distributed to hang indefinitely.
    for key in ("no_proxy", "http_proxy", "https_proxy", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY")
        delete!(ENV, key)
    end

    device = Reactant.XLA.default_device(Reactant.XLA.default_backend())
    client = Reactant.XLA.platform_name(Reactant.XLA.client(device))
    if client == "cuda"
        # This currently relies on <https://github.com/openxla/xla/pull/24689>. Hopefully in
        # the future there will be a better way to do this.
        if rendezvous_warn isa Int
            unsafe_store!(cglobal((:XLA_FIRST_CALL_RENDEZVOUS_WARN, libReactantExtra), Cint), rendezvous_warn)
        end
        if rendezvous_terminate isa Int
            unsafe_store!(cglobal((:XLA_FIRST_CALL_RENDEZVOUS_TERMINATE, libReactantExtra), Cint), rendezvous_terminate)
        end
    end
end
