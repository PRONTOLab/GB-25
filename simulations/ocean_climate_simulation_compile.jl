using GordonBell25: first_time_step!, loop!, try_compile_code, preamble, TRY_COMPILE_FAILED
using GordonBell25: data_free_ocean_climate_model_init, PROFILE
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState

PROFILE[] = true

preamble()

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState())

GC.gc(true); GC.gc(false); GC.gc(true)

TRY_COMPILE_FAILED[] = false
Ninner = ConcreteRNumber(2)

for optimize in (:before_raise, false, :before_jit), code_type in (:hlo, :xla)
    # We only want the optimised XLA code
    optimize in (:before_raise, false) && code_type === :xla && continue
    kernel_type = optimize === :before_raise ? "before_raise" : (optimize === false ? "unoptimised" : "optimised")
    @info "Compiling $(kernel_type) $(code_type) kernels..."
    if code_type === :hlo
        first_code = try_compile_code() do
            @code_hlo optimize=optimize raise=true first_time_step!(model)
        end
        loop_code = try_compile_code() do
            @code_hlo optimize=optimize raise=true loop!(model, Ninner)
        end
    elseif code_type === :xla
        first_code = try_compile_code() do
            @code_xla raise=true first_time_step!(model)
        end
        loop_code = try_compile_code() do
            @code_xla raise=true loop!(model, Ninner)
        end
    end
    for name in ("first", "loop"), debug in (true, false)
        # No debug info for `@code_xla`
        code_type === :xla && debug && continue
        open("$(kernel_type)_baroclinic_instability_simulation_$(name)$(debug ? "_debug" : "").$(code_type == :xla ? "xla" : "mlir")", "w") do io
            show(IOContext(io, :debug => debug), (Base.@locals())[Symbol(name, "_code")])
        end
    end
end

if TRY_COMPILE_FAILED[]
    error("compilation failed")
end
