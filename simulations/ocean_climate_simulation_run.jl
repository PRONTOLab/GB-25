using GordonBell25: ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")

@info "Generating model..."
model = ocean_climate_model_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true Oceananigans.TimeSteppers.first_time_step!(model)
rloop! = @compile raise=true loop!(model, 2)

@info "Running 1+2=3 steps..."
Reactant.with_profiler("./") do
    rfirst!(model)
    rloop!(model, 2)
end
@info "Done!"
