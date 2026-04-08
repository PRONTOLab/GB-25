#=
Verification: does `data_free_ocean_climate_model_init` work end-to-end
under `Distributed{ReactantState}` on this machine?

Same shape as `sharded_ocean_climate_simulation_run.jl` but uses
`data_free_ocean_climate_model_init` (the analytic-init version that
the rest of the repo's sim scripts use), so we can isolate whether the
sharded `loop!` compile failure is specific to the new
`ocean_climate_model_init` constructor or is a pre-existing
upstream limitation that data_free shares.
=#

using Dates
@info "data_free sharded check: starting" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: first_time_step!, loop!, factors, is_distributed_env_present
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.preamble()

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps_data_free", string(now(UTC)))
Reactant.Compiler.WHILE_CONCAT[] = true

GordonBell25.initialize(; single_gpu_per_process=false)

devarch = Oceananigans.ReactantState()
arch    = devarch

Ndev = length(Reactant.devices())
Rx, Ry = factors(Ndev)
if Ndev > 1
    arch = Oceananigans.Distributed(devarch; partition = Partition(Rx, Ry, 1))
    rank = Reactant.Distributed.local_rank()
else
    rank = 0
end

@info "[$rank] Ndev=$Ndev Rx,Ry=$Rx,$Ry"

resolution = parse(Float64, get(ENV, "RESOLUTION", "4"))
Nz         = parse(Int,     get(ENV, "NZ",         "10"))

@info "[$rank] Building data_free model (resolution=$resolution, Nz=$Nz)..." now(UTC)
model = GordonBell25.data_free_ocean_climate_model_init(arch;
                                                        resolution = resolution,
                                                        Nz         = Nz)
@info "[$rank] data_free model built" now(UTC)

Ninner = if Ndev == 1
    ConcreteRNumber(2)
else
    ConcreteRNumber(2; sharding=Sharding.NamedSharding(arch.connectivity, ()))
end

compile_options = CompileOptions(;
    sync = true, raise = true,
    strip_llvm_debuginfo = true,
)

@info "[$rank] Compiling first_time_step!..." now(UTC)
rfirst! = @compile compile_options=compile_options first_time_step!(model)
@info "[$rank] Compiled first_time_step! OK" now(UTC)

@info "[$rank] Running first_time_step!..." now(UTC)
@time "[$rank] first time step" rfirst!(model)
@info "[$rank] first_time_step! returned" now(UTC)

@info "[$rank] Compiling loop!(model, $Ninner)..." now(UTC)
rloop! = @compile compile_options=compile_options loop!(model, Ninner)
@info "[$rank] Compiled loop! OK" now(UTC)

@info "[$rank] Running loop!..." now(UTC)
@time "[$rank] loop" rloop!(model, Ninner)
@info "[$rank] loop! returned" now(UTC)

@info "[$rank] data_free SHARDED CHECK PASSED" now(UTC)
