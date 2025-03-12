using Reactant
using Oceananigans

function loop!(model, Ninner)
    Δt = 1200 # 20 minutes
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace for _ = 2:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end
