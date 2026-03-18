using ArgParse

const args_settings = ArgParseSettings()
@add_arg_table! args_settings begin
    "--grid-x"
        help = "Base factor for number of grid points on the x axis."
        default = 1536
        arg_type = Int
    "--grid-y"
        help = "Base factor for number of grid points on the y axis."
        default = 768
        arg_type = Int
    "--grid-z"
        help = "Base factor for number of grid points on the z axis."
        default = 4
        arg_type = Int
    "--float-type"
        help = "The default Oceananigans float type"
        default = "Float64"
        arg_type = String
end
const parsed_args = parse_args(ARGS, args_settings)

using BFloat16s
using GordonBell25: first_time_step!, loop!, try_compile_code, preamble, TRY_COMPILE_FAILED
using GordonBell25: baroclinic_instability_model, PROFILE, GordonBell25
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
Reactant.Compiler.WHILE_CONCAT[] = true

PROFILE[] = true
Oceananigans.defaults.FloatType = if parsed_args["float-type"] ∈ ("Float64", "f64")
    Float64
elseif parsed_args["float-type"] ∈ ("Float32", "f32")
    Float32
elseif parsed_args["float-type"] ∈ ("Float16", "f16")
    Float16
elseif parsed_args["float-type"] ∈ ("BFloat16", "bf16")
    Core.BFloat16
else
    throw(AssertionError("Unknown float type $(parsed_args["float-type"])"))
end

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

H = 8
Tx = parsed_args["grid-x"] * Rx
Ty = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nx = Tx - 2H
Ny = Ty - 2H

@info "Generating model..."
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@show model

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
        # No debug info for `@code_xla`
        code_type === :xla && debug && continue
        open("$(kernel_type)_sharded_baroclinic_instability_simulation_$(name)$(debug ? "_debug" : "").$(code_type == :xla ? "xla" : "mlir")", "w") do io
            show(IOContext(io, :debug => debug), (Base.@locals())[Symbol(name, "_code")])
        end
    end
end

if TRY_COMPILE_FAILED[]
    error("compilation failed")
end
