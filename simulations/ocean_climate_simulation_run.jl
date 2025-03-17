using GordonBell25: ocean_climate_simulation_run
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")

@info "Generating model..."
model = ocean_climate_simulation_run(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rloop! = @compile raise=true loop!(model, 2)

@info "Running..."
Reactant.with_profiler("./") do
    rloop!(model, 2)
end
@info "Done!"
