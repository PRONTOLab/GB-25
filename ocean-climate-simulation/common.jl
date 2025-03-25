using Reactant
using Oceananigans

function first_time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

time_step!(model) = Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)

function loop!(model, Nt)
    @trace for n = 1:Nt
        Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)
    end
    return nothing
end

Ninner = ConcreteRNumber(2)

# If we are in GitHub Actions, make `TMPDIR` be a local directory from which we
# can upload artifacts at the end.
if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "tmp"))
end

