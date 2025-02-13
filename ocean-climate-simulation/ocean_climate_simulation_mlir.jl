ENV["dont-run"] = true
ENV["raise"] = true
ENV["use-reactant"] = true

include("ocean_climate_simulation.jl")

unopt = @code_hlo optimize=false run!(simulation) 

# Unoptimized HLO
open("unopt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(unopt))
end

# Optimized HLO
opt = @code_hlo optimize=:before_jit run!(simulation) 

open("opt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(opt))
end

