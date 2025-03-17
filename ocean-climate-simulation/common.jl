using Reactant
using Oceananigans
 
const Δt = 30 # seconds
    
first_time_step!(model) = Oceananigans.TimeSteppers.first_time_step!(model, Δt)

function loop!(model, Ninner)
    Δt = 30 # seconds
    @trace for _ = 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

