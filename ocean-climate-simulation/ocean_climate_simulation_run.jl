using GordonBell25: data_free_ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rloop! = @compile raise=true loop!(model, 2)

@info "Running..."
Reactant.with_profiler("./") do
    rloop!(model, 2)
end
@info "done!"
