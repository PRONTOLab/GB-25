using GordonBell25: data_free_ocean_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

@info "Generating model..."
model = data_free_ocean_model_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true first_time_step!(model)
rloop! = @compile raise=true loop!(model, Ninner)

Reactant.with_profiler("./") do
    rfirst!(model)
    rloop!(model, Ninner)
end
@info "Done!"

