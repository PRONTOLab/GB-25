using GordonBell25
using GordonBell25: first_time_step!, loop!, try_compile_code, preamble, TRY_COMPILE_FAILED
using GordonBell25: moist_baroclinic_wave_model, PROFILE
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 16,
)

default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type

using CUDA
using Reactant
using Oceananigans.Architectures: ReactantState

PROFILE[] = true

preamble()

H = 8
Nλ = parsed_args["grid-x"] - 2H
Nφ = parsed_args["grid-y"] - 2H
Nz = parsed_args["grid-z"]

@info "Generating atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz)..."
arch = ReactantState()
model = moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz, Δt=2.0, halo=(H, H, H))

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
