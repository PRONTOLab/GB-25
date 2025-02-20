using Profile

ENV["run"] = false
ENV["raise"] = true
ENV["use-reactant"] = true

include("data_free_ocean_climate_simulation.jl")

Profile.clear()
# @profile
unopt = @code_hlo optimize=false run!(simulation)
# open("profile_unopt_ocean_climate_simulation.txt", "w") do s
#     Profile.print(IOContext(s, :displaysize => (48, 1000)))
# end

# Unoptimized HLO
open("unopt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(unopt))
end

# Optimized HLO
Profile.clear()
#@profile
opt = @code_hlo optimize=:before_jit run!(simulation)
# open("profile_opt_ocean_climate_simulation.txt", "w") do s
#     Profile.print(IOContext(s, :displaysize => (48, 1000)))
# end

open("opt_ocean_climate_simulation.mlir", "w") do io
    write(io, string(opt))
end
