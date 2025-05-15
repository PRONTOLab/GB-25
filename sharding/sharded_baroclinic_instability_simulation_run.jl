using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using Reactant
Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true

jobid_procid = GordonBell25.get_jobid_procid()

# This must be called before `GordonBell25.initialize`!
GordonBell25.preamble()

using Libdl: dllist
@show filter(contains("nccl"), dllist())

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
# Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
# Reactant.Compiler.DUS_TO_CONCAT[] = false
# Reactant.Compiler.SUM_TO_REDUCEWINDOW[] = true
# Reactant.Compiler.AGGRESSIVE_SUM_TO_CONV[] = true

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

Rx, Ry = factors(Ndev)
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

@info "[$rank] allocations" GordonBell25.allocatorstats()
H = 8
Tx = 32 * 48 * Rx
Ty = 32 * 24 * Ry
Nz = 4

Nx = Tx - 2H
Ny = Ty - 2H

@info "[$rank] Generating model..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@info "[$rank] allocations" GordonBell25.allocatorstats()

@show model

Ninner = ConcreteRNumber(256; sharding=Sharding.NamedSharding(arch.connectivity, ()))

@info "[$rank] Compiling first_time_step!..." now(UTC)
rfirst! = @compile sync=true raise=true first_time_step!(model)
@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] Compiling loop..." now(UTC)
compiled_loop! = @compile sync=true raise=true loop!(model, Ninner)
@info "[$rank] allocations" GordonBell25.allocatorstats()

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)
mkpath(joinpath(profile_dir, "first_time_step"))
@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] Running first_time_step!..." now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "first_time_step")) do
    @time "[$rank] first time step" rfirst!(model)
end
@info "[$rank] allocations" GordonBell25.allocatorstats()

mkpath(joinpath(profile_dir, "loop"))
@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] running loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop")) do
    @time "[$rank] loop" compiled_loop!(model, Ninner)
end

mkpath(joinpath(profile_dir, "loop 2"))
@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] running second loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop")) do
    @time "[$rank] second loop" compiled_loop!(model, Ninner)
end
@info "[$rank] allocations" GordonBell25.allocatorstats()

@info "Done!" now(UTC)
