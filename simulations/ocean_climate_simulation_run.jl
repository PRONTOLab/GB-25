using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

using InteractiveUtils

using Oceananigans

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: update_state!, time_step!

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState(); resolution=4, Nz=10)
Ninner = ConcreteRNumber(2)

GC.gc(true); GC.gc(false); GC.gc(true)

Δt = model.clock.last_Δt

function my_first_time_step!(model, Δt)
    initialize!(model)
    # The first update_state is conditionally gated from within time_step! normally, but not Reactant
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

@info "Compiling..."
@show @which initialize!(model)
@show @which update_state!(model)
@show @which time_step!(model)
rfirst! = @compile raise=true sync=true my_first_time_step!(model, Δt)

@info "Done!"
