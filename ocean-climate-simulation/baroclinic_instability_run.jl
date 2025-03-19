using GordonBell25: baroclinic_instability_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")

@info "Generating model..."
Δt = 60 # seconds
model = baroclinic_instability_model_init(ReactantState(), Δt=60, resolution=1/4)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true first_time_step!(model)
#rloop! = @compile raise=true loop!(model, Ninner)

@info "Running..."
Reactant.with_profiler("./") do
    rfirst!(model)
    #rloop!(model, Ninner)
end
@info "Done!"

