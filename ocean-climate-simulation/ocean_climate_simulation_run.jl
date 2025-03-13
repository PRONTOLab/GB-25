using GordonBell25: data_free_ocean_climate_simulation_init
using Oceananigans: run!
using Oceananigans.Architectures: ReactantState
using Reactant

simulation = data_free_ocean_climate_simulation_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

Reactant.with_profiler("./") do
    @jit raise=true run!(simulation)
end
