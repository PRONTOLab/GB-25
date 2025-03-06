using GordonBell25: data_free_ocean_climate_simulation_init
using Reactant: @code_hlo
using Oceananigans: run!


simulation = data_free_ocean_climate_simulation_init()

unopt = @code_hlo optimize=false raise=true run!(simulation)

# Unoptimized HLO
open("unopt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(unopt))
end

# Optimized HLO
opt = @code_hlo optimize=:before_jit raise=true run!(simulation)

open("opt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(opt))
end
