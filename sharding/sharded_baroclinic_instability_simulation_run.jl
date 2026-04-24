using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using BFloat16s
using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors, is_distributed_env_present

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 1536,
    grid_y_default = 768,
    grid_z_default = 4,
)

using Oceananigans

Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

jobid_procid = GordonBell25.get_jobid_procid()

# This must be called before `GordonBell25.initialize`!
GordonBell25.preamble()

using Libdl: dllist
@show filter(contains("nccl"), dllist())

const model_state_dump_path = joinpath(@__DIR__, "model_dumps", jobid_procid)

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
# Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
# Reactant.Compiler.DUS_TO_CONCAT[] = false
# Reactant.Compiler.SUM_TO_REDUCEWINDOW[] = true
# Reactant.Compiler.AGGRESSIVE_SUM_TO_CONV[] = true

GordonBell25.initialize(; single_gpu_per_process=false)

# devarch = Oceananigans.GPU()
devarch = Oceananigans.ReactantState()

arch = devarch

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
    arch = Oceananigans.Distributed(
	arch;
        partition = Partition(Rx, Ry, 1)
    )
    rank = if devarch isa Oceananigans.ReactantState
	Reactant.Distributed.local_rank()
    else
       comm = MPI.COMM_WORLD
       MPI.Comm_rank(comm)
    end
end

@info "[$rank] allocations" GordonBell25.allocatorstats()
H = 8
Tx = parsed_args["grid-x"] * Rx
Ty = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nx = Tx - 2H
Ny = Ty - 2H

@info "[$rank] Generating model (Nx=$Nx, Ny=$Ny)..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@info "[$rank] allocations" GordonBell25.allocatorstats()

@show model

Ninner = 256

if devarch isa Oceananigans.ReactantState
   Ninner = if Ndev == 1
	 ConcreteRNumber(Ninner)
   else
   	ConcreteRNumber(Ninner; sharding=Sharding.NamedSharding(arch.connectivity, ()))
   end
end

@info "[$rank] Compiling first_time_step!..." now(UTC)

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all, multifloat=GordonBell25.multifloat_from_args(parsed_args))
# # uncomment to turn communication optimizations off
# # Note: may need to also increase xla timeout to get this to run, eg:
# # # export XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=100 --xla_gpu_first_collective_call_terminate_timeout_seconds=300 \${XLA_FLAGS}"
# compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all, multifloat=GordonBell25.multifloat_from_args(parsed_args), xla_debug_options=(xla_enable_enzyme_comms_opt=false,), optimize_communications=false)

rfirst! = if devarch isa Oceananigans.ReactantState
     @compile compile_options=compile_options first_time_step!(model)
else
     first_time_step!     
end

@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] Compiling loop..." now(UTC)

compiled_loop! = if devarch isa Oceananigans.ReactantState
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

let dump_path = mkpath(joinpath(model_state_dump_path, "loop1"))
    @info "[$rank] loop1 dumping state to disk" now(UTC) dump_path
    GordonBell25.save_model_state(dump_path, model, arch)
    @info "[$rank] loop1 successfully dumped to disk" now(UTC)
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

let dump_path = mkpath(joinpath(model_state_dump_path, "loop2"))
    @info "[$rank] loop2 dumping state to disk" now(UTC) dump_path
    GordonBell25.save_model_state(dump_path, model, arch)
    @info "[$rank] loop2 successfully dumped to disk" now(UTC)
end

@info "[$rank] Done!" now(UTC)
