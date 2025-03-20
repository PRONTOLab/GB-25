using Reactant
using Oceananigans

function first_time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

function loop!(model, Ninner)
    Δt = model.clock.last_Δt
    @trace for _ = 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

# If we are in GitHub Actions, make `TMPDIR` be a local directory from which we
# can upload artifacts at the end.
if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "tmp"))
end

