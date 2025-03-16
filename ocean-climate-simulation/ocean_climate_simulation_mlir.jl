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

# Pre-raise IR
@info "Compiling before raise kernel..."
before_raise = try
    @code_hlo optimize=:before_raise raise=true loop!(model, ConcreteRNumber(2))
catch e
    @error "Failed to compile" exception=(e, catch_backtrace())
    Text("""
    // Failed to compile
    //$e
    """)
end

# Unoptimized HLO
@info "Compiling unoptimised kernel..."
unopt = try
    @code_hlo optimize=false raise=true loop!(model, ConcreteRNumber(2))
catch e
    @error "Failed to compile" exception=(e, catch_backtrace())
    Text("""
    // Failed to compile
    //$e
    """)
end

# Optimized HLO
@info "Compiling optimised kernel..."
opt = try
    @code_hlo optimize=:before_jit raise=true loop!(model, ConcreteRNumber(2))
catch e
    @error "Failed to compile" exception=(e, catch_backtrace())
    Text("""
    // Failed to compile
    //$e
    """)
end

for debug in (true, false)
    open("before_raise_ocean_climate_simulation$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), unopt)
    end

    # Unoptimized HLO
    open("unopt_ocean_climate_simulation$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), unopt)
    end

    open("opt_ocean_climate_simulation$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), opt)
    end
end
