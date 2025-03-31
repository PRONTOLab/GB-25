using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

preamble()

Ninner = ConcreteRNumber(3)

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState())
Ninner = ConcreteRNumber(2)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true sync=true first_time_step!(model)
rstep! = @compile raise=true sync=true time_step!(model)
rloop! = @compile raise=true sync=true loop!(model, Ninner)

@info "Running..."
Reactant.with_profiler("./") do
    rfirst!(model)
end
Reactant.with_profiler("./") do
    rstep!(model)
end
Reactant.with_profiler("./") do
    rloop!(model, Ninner)
end
@info "Done!"
