using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: baroclinic_instability_model
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

preamble()

Ninner = ConcreteRNumber(3)
Oceananigans.defaults.FloatType = Float32

@info "Generating model..."
arch = ReactantState()
#arch = Distributed(ReactantState(), partition=Partition(2, 2, 1))
model = baroclinic_instability_model(arch, resolution=8, Δt=60, Nz=10)

GC.gc(true); GC.gc(false); GC.gc(true)

using InteractiveUtils

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: update_state!, time_step!


Δt = model.clock.last_Δt

function my_first_time_step!(model, Δt)
    #initialize!(model)
    update_state!(model)
    #time_step!(model, Δt, euler=true)
    return nothing
end

@show @which my_first_time_step!(model, Δt)

@info "Compiling..."
rfirst! = @compile raise=true sync=true my_first_time_step!(model, Δt)
