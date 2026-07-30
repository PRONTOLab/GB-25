using ArgParse

const args_settings = ArgParseSettings()
@add_arg_table! args_settings begin
    "--grid-x"
        help = "Base factor for number of grid points on the x axis."
        default = 64
        arg_type = Int
    "--grid-y"
        help = "Base factor for number of grid points on the y axis."
        default = 64
        arg_type = Int
    "--grid-z"
        help = "Base factor for number of grid points on the z axis."
        default = 4
        arg_type = Int
end
const parsed_args = parse_args(ARGS, args_settings)

using GordonBell25: first_time_step!, loop!, try_compile_code, preamble, TRY_COMPILE_FAILED
using GordonBell25: data_free_ocean_climate_model_init, PROFILE, GordonBell25
using Reactant
using Oceananigans
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

H = 8
Tx = parsed_args["grid-x"] * Rx
Ty = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nx = Tx - 2H
Ny = Ty - 2H

grid_type = Symbol(get(ENV, "grid_type", "simple_lat_lon"))
@info "Generating model (grid_type=$grid_type)..."
model = data_free_ocean_climate_model_init(arch, Nx, Ny, Nz; halo=(H, H, H), grid_type,
                                          set_initial_conditions=false)
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
        open("$(kernel_type)_sharded_ocean_climate_simulation_$(name)$(debug ? "_debug" : "").$(code_type == :xla ? "xla" : "mlir")", "w") do io
            show(IOContext(io, :debug => debug), (Base.@locals())[Symbol(name, "_code")])
        end
    end
end

if TRY_COMPILE_FAILED[]
    error("compilation failed")
end
