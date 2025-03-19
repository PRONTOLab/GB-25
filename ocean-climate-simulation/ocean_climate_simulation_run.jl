using GordonBell25: data_free_ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rloop! = @compile raise=true sync=true loop!(model, Ninner)

@info "Running..."
Reactant.with_profiler("./") do
    rloop!(model, Ninner)
end
@info "Done!"
