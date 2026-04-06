using GordonBell25: first_time_step!, loop!, try_compile_code, preamble, TRY_COMPILE_FAILED
using GordonBell25: moist_baroclinic_wave_model, PROFILE, GordonBell25
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
Reactant.Compiler.WHILE_CONCAT[] = true

PROFILE[] = true

preamble()

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

Rx, Ry = GordonBell25.factors(Ndev)
if Ndev == 1
    rank = 0
    arch = Oceananigans.ReactantState()
else
    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(Rx, Ry, 1)
    )
    rank = Reactant.Distributed.local_rank()
end

Nλ = parsed_args["grid-x"] * Rx
Nφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

@info "Generating atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz)..."
model = moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz, Δt=2.0, halo=(8, 8, 8))
@show model

GC.gc(true); GC.gc(false); GC.gc(true)

TRY_COMPILE_FAILED[] = false
Ninner = ConcreteRNumber(2)

for optimize in (:before_raise, false, :before_jit), code_type in (:hlo, :xla)
    optimize in (:before_raise, false) && code_type === :xla && continue
    kernel_type = optimize === :before_raise ? "before_raise" : (optimize === false ? "unoptimised" : "optimised")
    @info "Compiling $(kernel_type) $(code_type) kernels..."
    if code_type === :hlo
        first_code = try_compile_code() do
            @code_hlo optimize=optimize raise=true shardy_passes=:post_sdy_propagation first_time_step!(model)
        end
        loop_code = try_compile_code() do
            @code_hlo optimize=optimize raise=true shardy_passes=:post_sdy_propagation loop!(model, Ninner)
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
        open("$(kernel_type)_sharded_atmosphere_simulation_$(name)$(debug ? "_debug" : "").$(code_type == :xla ? "xla" : "mlir")", "w") do io
            show(IOContext(io, :debug => debug), (Base.@locals())[Symbol(name, "_code")])
        end
    end
end

if TRY_COMPILE_FAILED[]
    error("compilation failed")
end
