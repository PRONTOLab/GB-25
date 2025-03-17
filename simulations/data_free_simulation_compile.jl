using GordonBell25: data_free_ocean_model_init, PROFILE
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState

PROFILE[] = true

include("common.jl")

@info "Generating model..."
model = data_free_ocean_model_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

failed = false

# Pre-raise IR
@info "Compiling before raise kernel..."
before_raise = try
    @code_hlo optimize=:before_raise raise=true loop!(model, 2)
catch e
    @error "Failed to compile" exception=(e, catch_backtrace())
    global failed = true
    Text("""
    // Failed to compile
    //$e
    """)
end

# Unoptimized HLO
@info "Compiling unoptimised kernel..."
unopt = try
    @code_hlo optimize=false raise=true loop!(model, 2)
catch e
    @error "Failed to compile" exception=(e, catch_backtrace())
    global failed = true
    Text("""
    // Failed to compile
    //$e
    """)
end

# Optimized HLO
@info "Compiling optimised kernel..."
opt = try
    @code_hlo optimize=:before_jit raise=true loop!(model, 2)
catch e
    @error "Failed to compile" exception=(e, catch_backtrace())
    global failed = true
    Text("""
    // Failed to compile
    //$e
    """)
end

for debug in (true, false)
    open("before_raise_data_free$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), unopt)
    end

    # Unoptimized HLO
    open("unopt_ocean_data_free$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), unopt)
    end

    open("opt_data_free$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), opt)
    end
end

if failed
    error("compilation failed")
end
