using Reactant
using Oceananigans.Architectures: Architectures

include("data_free_ocean_climate_model.jl")
PROFILE[] = true

@info "Generating model..."
model = data_free_ocean_climate_model_init(Architectures.ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

function loop!(model)
    Δt = 1200 # 20 minutes
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace for _ = 2:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

# Unoptimized HLO
@info "Compiling unoptimised kernel..."
unopt = @code_hlo optimize=false raise=true loop!(model)

open("unopt_ocean_climate_simulation.mlir", "w") do io
    show(io, unopt)
end

# Optimized HLO
@info "Compiling optimised kernel..."
opt = @code_hlo optimize=:before_jit raise=true loop!(model)

open("opt_ocean_climate_simulation.mlir", "w") do io
    show(io, opt)
end
