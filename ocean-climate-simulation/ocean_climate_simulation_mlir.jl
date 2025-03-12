using Reactant
using Oceananigans.Architectures: Architectures

include("data_free_ocean_climate_simulation.jl")
PROFILE[] = true
simulation = data_free_ocean_climate_simulation_init(Architectures.ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

unopt = @code_hlo optimize=false raise=true run!(simulation)

# Unoptimized HLO
open("unopt_ocean_climate_simulation.mlir", "w") do io
    show(IOContext(io, :debug => true), unopt)
end

# Optimized HLO
opt = @code_hlo optimize=:before_jit raise=true run!(simulation)

open("opt_ocean_climate_simulation.mlir", "w") do io
    show(IOContext(io, :debug => true), opt)
end
