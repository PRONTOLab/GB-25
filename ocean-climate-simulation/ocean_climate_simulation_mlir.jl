using Dates
@info "loading GB25" now(UTC)
using GordonBell25: data_free_ocean_climate_simulation_init, PROFILE
@info "loading Reactant" now(UTC)
using Reactant: @code_hlo
@info "loading Reactant" now(UTC)
using Oceananigans: run!
using Oceananigans.Architectures: ReactantState

# PROFILE[] = true
@info "init" now(UTC)
simulation = data_free_ocean_climate_simulation_init(ReactantState())

@info "all done" now(UTC)

# GC.gc(true); GC.gc(false); GC.gc(true)

# unopt = @code_hlo optimize=false raise=true run!(simulation)

# # Unoptimized HLO
# open("unopt_ocean_climate_simulation.mlir", "w") do io
#     show(io, unopt)
# end

# # Optimized HLO
# opt = @code_hlo optimize=:before_jit raise=true run!(simulation)

# open("opt_ocean_climate_simulation.mlir", "w") do io
#     show(io, opt)
# end
