using BFloat16s
using GordonBell25: first_time_step!, loop!, try_compile_code, preamble, TRY_COMPILE_FAILED
using GordonBell25: baroclinic_instability_model, PROFILE, GordonBell25, is_distributed_env_present
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
Reactant.Compiler.WHILE_CONCAT[] = true

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 1536,
    grid_y_default = 768,
    grid_z_default = 4,
)

PROFILE[] = true
Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)

if !is_distributed_env_present()
    using MPI
    MPI.Init()
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

for optimize in (:before_raise, false, :before_jit, true), code_type in (:hlo, :xla)
    # We only want the optimised XLA code
    optimize in (:before_raise, false, :before_jit) && code_type === :xla && continue
    optimize == true && code_type !== :xla && continue
    kernel_type = optimize isa Bool ? (optimize === false ? "unoptimised" : "optimised") : string(optimize)

    compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=["enzymexla.kernel_call", "(::Reactant.Compiler.LLVMFunc", "ka_with_reactant", "(::KernelAbstractions.Kernel", "var\"#_launch!;_launch!"], multifloat=GordonBell25.multifloat_from_args(parsed_args), optimization_passes=optimize)
    @info "Compiling $(kernel_type) $(code_type) kernels..."
    if code_type === :hlo
        first_code = try_compile_code() do
            @code_hlo compile_options=compile_options first_time_step!(model)
        end
        loop_code = try_compile_code() do
            @code_hlo compile_options=compile_options loop!(model, Ninner)
        end
    elseif code_type === :xla
        first_code = try_compile_code() do
            @code_xla compile_options=compile_options first_time_step!(model)
        end
        loop_code = try_compile_code() do
            @code_xla compile_options=compile_options loop!(model, Ninner)
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
