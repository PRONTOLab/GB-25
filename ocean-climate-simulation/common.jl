using Reactant
using Oceananigans

const Δt = 1200 # 20 minutes

first_time_step!(model) = Oceananigans.TimeSteppers.first_time_step!(model, Δt)

function loop!(model, Ninner)
    @trace for _ = 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end
