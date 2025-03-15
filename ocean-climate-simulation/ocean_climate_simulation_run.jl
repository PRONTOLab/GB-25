using GordonBell25: data_free_ocean_climate_simulation_init
using Oceananigans: run!
using Oceananigans.Architectures: ReactantState
using Reactant

Reactant.Compiler.SROA_ATTRIBUTOR[] = false

@info "Generating model..."
simulation = data_free_ocean_climate_simulation_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rrun! = @compile raise=true run!(simulation)

@info "Running..."
Reactant.with_profiler("./") do
    rrun!(simulation)
end
@info "Done!"
