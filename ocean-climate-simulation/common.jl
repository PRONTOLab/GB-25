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

    #Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end

