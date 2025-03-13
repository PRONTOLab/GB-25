using GordonBell25: data_free_ocean_climate_model_init, PROFILE
using Reactant: @code_hlo, @trace
using Oceananigans
using Oceananigans.Architectures: ReactantState

PROFILE[] = true

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

function loop!(model, Ninner)
    Δt = 1200 # 20 minutes
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace for _ = 2:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

# Unoptimized HLO
@info "Compiling unoptimised kernel..."
unopt = @code_hlo optimize=false raise=true loop!(model, 10)

# Optimized HLO
@info "Compiling optimised kernel..."
opt = @code_hlo optimize=:before_jit raise=true loop!(model, 10)

for debug in (true, false)
    # Unoptimized HLO
    open("unopt_ocean_climate_simulation$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), unopt)
    end

    open("opt_ocean_climate_simulation$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), opt)
    end
end
