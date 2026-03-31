using Dates
@info "This is when the fun begins" now(UTC)

using ArgParse

const args_settings = ArgParseSettings()
@add_arg_table! args_settings begin
    "--grid-x"
        help = "Base factor for number of grid points on the λ axis."
        default = 48
        arg_type = Int
    "--grid-y"
        help = "Base factor for number of grid points on the φ axis."
        default = 24
        arg_type = Int
    "--grid-z"
        help = "Number of grid points on the z axis."
        default = 10
        arg_type = Int
    "--precision"
        help = "Number of bits of precision"
        default = 64
        arg_type = Int
end
const parsed_args = parse_args(ARGS, args_settings)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors, is_distributed_env_present
using Oceananigans
if parsed_args["precision"] == 64
    Oceananigans.defaults.FloatType = Float64
elseif parsed_args["precision"] == 32
    Oceananigans.defaults.FloatType = Float32
else
    throw(AssertionError("Unknown precision $(parsed_args["precision"])"))
end
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using CUDA
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

jobid_procid = GordonBell25.get_jobid_procid()

GordonBell25.preamble()

using Libdl: dllist
@show filter(contains("nccl"), dllist())

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.WHILE_CONCAT[] = true

GordonBell25.initialize(; single_gpu_per_process=false)

local_arch = Oceananigans.ReactantState()
arch = local_arch

Ndev = if arch isa Oceananigans.ReactantState
   length(Reactant.devices())
else
   comm = MPI.COMM_WORLD
   MPI.Comm_size(comm)
end

@show Ndev

Rx, Ry = factors(Ndev)

if Ndev == 1
    rank = 0
else
    arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
    rank = if local_arch isa Oceananigans.ReactantState
	    Reactant.Distributed.local_rank()
    else
       comm = MPI.COMM_WORLD
       MPI.Comm_rank(comm)
    end
end

@info "[$rank] allocations" GordonBell25.allocatorstats()

Nλ = parsed_args["grid-x"] * Rx
Nφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

@info "[$rank] Generating atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz)..." now(UTC)
model = GordonBell25.moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz, Δt=2.0, halo=(8, 8, 8))
@info "[$rank] allocations" GordonBell25.allocatorstats()

@show model

Ninner = 256

if local_arch isa Oceananigans.ReactantState
    Ninner = if Ndev == 1
        ConcreteRNumber(Ninner)
    else
        sharding = Sharding.NamedSharding(arch.connectivity, ())
   	    ConcreteRNumber(Ninner; sharding)
    end
end

@info "[$rank] Compiling first_time_step!..." now(UTC)
compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=["enzymexla.kernel_call", "(::Reactant.Compiler.LLVMFunc", "ka_with_reactant", "(::KernelAbstractions.Kernel", "var\"#_launch!;_launch!"])
rfirst! = if local_arch isa Oceananigans.ReactantState
     @compile compile_options=compile_options first_time_step!(model)
else
     first_time_step!
end

@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] Compiling loop..." now(UTC)

compiled_loop! = if local_arch isa Oceananigans.ReactantState
     @compile compile_options=compile_options loop!(model, Ninner)
else
     loop!
end

@info "[$rank] allocations" GordonBell25.allocatorstats()

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)
mkpath(joinpath(profile_dir, "first_time_step"))
@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] Running first_time_step!..." now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "first_time_step")) do
    Reactant.Profiler.annotate("bench"; metadata=Dict("step_num" => 1, "_r" => 1)) do
        @time "[$rank] first time step" rfirst!(model)
    end
end
@info "[$rank] allocations" GordonBell25.allocatorstats()

mkpath(joinpath(profile_dir, "loop"))
@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] running loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop")) do
    Reactant.Profiler.annotate("bench"; metadata=Dict("step_num" => 1, "_r" => 1)) do
        @time "[$rank] loop" compiled_loop!(model, Ninner)
    end
end

mkpath(joinpath(profile_dir, "loop2"))
@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] running second loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop2")) do
    Reactant.Profiler.annotate("bench"; metadata=Dict("step_num" => 1, "_r" => 1)) do
        @time "[$rank] second loop" compiled_loop!(model, Ninner)
    end
end
@info "[$rank] allocations" GordonBell25.allocatorstats()

@info "[$rank] Done!" now(UTC)
