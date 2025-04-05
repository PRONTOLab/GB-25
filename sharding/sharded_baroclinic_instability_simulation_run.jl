# Unset environment variables which would cause XLA distributed to hang indefinitely.
for key in ("no_proxy", "http_proxy", "https_proxy", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY")
    delete!(ENV, key)
end

using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"
jobid_procid = string(get(ENV, "SLURM_JOB_ID", Int(datetime2unix(now(UTC)) * 1000)), ".", get(ENV, "SLURM_PROCID", string(getpid())))

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using Reactant

using Libdl: dllist
@show filter(contains("nccl"), dllist())

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
Reactant.Compiler.DUS_TO_CONCAT[] = true
# Reactant.DEBUG_ENSURE_ALWAYS_SHARDED[] = true

GordonBell25.initialize(; single_gpu_per_process=false)

ndevices = length(Reactant.devices())

process_id = Reactant.Distributed.local_rank()
arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition=Partition(factors(ndevices)..., 1)
)

Nz = 128

@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
model = GordonBell25.baroclinic_instability_model(arch; grid_type=:simple_lat_lon, Δt=1, Nz,
                                                  resolution=1/0.25)
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

@show model

Ninner = ConcreteRNumber(256; sharding=Sharding.NamedSharding(arch.connectivity, ()))

@info "[$(process_id)] Compiling first_time_step!..."
rfirst! = @compile sync=true raise=true first_time_step!(model)
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
@info "[$(process_id)] Compiling loop..."
compiled_loop! = @compile sync=true raise=true loop!(model, Ninner)
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)
mkpath(joinpath(profile_dir, "first_time_step"))
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
@info "[$(process_id)] running first time step" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "first_time_step")) do
    @time "[$(process_id)] first time step" rfirst!(model)
end
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

mkpath(joinpath(profile_dir, "loop"))
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
@info "[$(process_id)] running loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop")) do
    @time "[$(process_id)] loop" compiled_loop!(model, Ninner)
end
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
