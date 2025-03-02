ENV["run"] = false
ENV["use-reactant"] = true

Reactant.Compiler.DUMP_LLVMIR[] = true

include("data_free_ocean_climate_simulation.jl")

unopt = @code_hlo optimize=false raise=false run!(simulation)

# Unoptimized HLO
open("unopt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(unopt))
end

# Optimized HLO
opt = @code_hlo optimize=:before_jit raise=false run!(simulation)

open("opt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(opt))
end
