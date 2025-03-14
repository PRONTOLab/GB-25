using GordonBell25: data_free_ocean_climate_simulation_init, PROFILE
using Reactant: @code_hlo
using Oceananigans: run!
using Oceananigans.Architectures: ReactantState

PROFILE[] = true
simulation = data_free_ocean_climate_simulation_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

unopt = @code_hlo optimize=false raise=true run!(simulation)

# Optimized HLO
opt = @code_hlo optimize=:before_jit raise=true run!(simulation)

for debug in (true, false)
    # Unoptimized HLO
    open("unopt_ocean_climate_simulation$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), unopt)
    end

    open("opt_ocean_climate_simulation$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), opt)
    end
end
