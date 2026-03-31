using GordonBell25: first_time_step!, loop!, try_compile_code, preamble, TRY_COMPILE_FAILED
using GordonBell25: moist_baroclinic_wave_model, set_moist_baroclinic_wave!, PROFILE
using CUDA
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState

PROFILE[] = true

preamble()

@info "Generating atmosphere model..."
arch = ReactantState()
model = moist_baroclinic_wave_model(arch; Nλ=48, Nφ=24, Nz=10, Δt=2.0)

@info "Setting initial conditions..."
set_moist_baroclinic_wave!(model)

GC.gc(true); GC.gc(false); GC.gc(true)

TRY_COMPILE_FAILED[] = false
Ninner = ConcreteRNumber(2)

for optimize in (:before_raise, false, :before_jit), code_type in (:hlo, :xla)
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
        code_type === :xla && debug && continue
        open("$(kernel_type)_atmosphere_simulation_$(name)$(debug ? "_debug" : "").$(code_type == :xla ? "xla" : "mlir")", "w") do io
            show(IOContext(io, :debug => debug), (Base.@locals())[Symbol(name, "_code")])
        end
    end
end

if TRY_COMPILE_FAILED[]
    error("compilation failed")
end
