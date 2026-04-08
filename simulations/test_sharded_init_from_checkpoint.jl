#=
Test that `initial_conditions_path` works under sharded Reactant.

Builds the 1/4° baroclinic instability model with
`arch = Distributed(ReactantState(); partition=Partition(Rx, Ry, 1))`,
loads T, S from the cached JLD2 artifact via the `initial_conditions_path`
keyword (which goes through the load → CPU twin → on-arch source Field →
`interpolate!` workflow), then compiles and runs `first_time_step!` and a
short `loop!` to verify everything composes under Reactant.

Run (4 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
        julia --project -O0 simulations/test_sharded_init_from_checkpoint.jl
=#

using Dates
using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors, is_distributed_env_present
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

const CHECKPOINT = joinpath(@__DIR__, "initial_conditions", "baroclinic_ic_quarter_degree.jld2")
const Nx, Ny, Nz = 1536, 768, 64
const Δt = 4minutes

GordonBell25.preamble()
GordonBell25.initialize(; single_gpu_per_process=false)

Ndev = length(Reactant.devices())
Rx, Ry = factors(Ndev)
arch = Oceananigans.Distributed(ReactantState(); partition = Partition(Rx, Ry, 1))
rank = Reactant.Distributed.local_rank()

@info "[sharded-reactant] Building model from checkpoint" CHECKPOINT Ndev Rx Ry now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz;
                                                  halo=(8, 8, 8), Δt=Δt,
                                                  initial_conditions_path = CHECKPOINT)
@info "[$rank] Model built" now(UTC)

# Sanity: pull T/S to host and confirm non-zero. Each rank's `Array(interior(...))`
# triggers an XLA all-gather to host (acceptable for a small test).
T_host = Array(interior(model.tracers.T))
S_host = Array(interior(model.tracers.S))
@info "[$rank] T/S after init" extrema_T=extrema(T_host) extrema_S=extrema(S_host) mean_T=sum(T_host)/length(T_host)

Ninner = ConcreteRNumber(10; sharding=Sharding.NamedSharding(arch.connectivity, ()))

compile_options = CompileOptions(;
    sync=true, raise=true, strip_llvm_debuginfo=true,
    strip=["enzymexla.kernel_call", "(::Reactant.Compiler.LLVMFunc",
           "ka_with_reactant", "(::KernelAbstractions.Kernel",
           "var\"#_launch!;_launch!"],
)

@info "[$rank] Compiling first_time_step!..." now(UTC)
rfirst! = @compile compile_options=compile_options first_time_step!(model)

@info "[$rank] Compiling loop!..." now(UTC)
rloop! = @compile compile_options=compile_options loop!(model, Ninner)

@info "[$rank] Running first_time_step!..." now(UTC)
@time "[$rank] first time step" rfirst!(model)

@info "[$rank] Running loop (10 steps)..." now(UTC)
@time "[$rank] loop10" rloop!(model, Ninner)

@info "[$rank] Final clock" time=model.clock.time iteration=model.clock.iteration
@info "[$rank] DONE" now(UTC)
