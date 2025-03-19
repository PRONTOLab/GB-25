using Reactant
using Oceananigans
    
function first_time_step!(model)
    Δt = model.clock.last_Δt # note: fixed on initialization
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

function loop!(model, Ninner)
    Δt = model.clock.last_Δt # note: fixed on initialization
    @trace for _ = 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

Ninner = ConcreteRNumber(2)
