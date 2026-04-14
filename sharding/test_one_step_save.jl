using Dates
@info "Test: one time step + save (JLD2)" now(UTC)

using GordonBell25
using GordonBell25: first_time_step!, loop!, factors, is_distributed_env_present
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64, grid_y_default = 64, grid_z_default = 64)

Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)

using Oceananigans.Architectures: ReactantState
using Printf
using CUDA
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.preamble()
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = false
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
GordonBell25.initialize(; single_gpu_per_process=false)

local_arch = ReactantState()
arch = local_arch
Ndev = length(Reactant.devices())
Rx, Ry = factors(Ndev)
if Ndev == 1; rank = 0
else
    arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
    rank = Reactant.Distributed.local_rank()
end

H = 4
Nλ = parsed_args["grid-x"] * Rx - 2H
Nφ = parsed_args["grid-y"] * Ry - 2H
Nz = parsed_args["grid-z"]

@info "[$rank] Building model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz)..." now(UTC)
model = GordonBell25.moist_baroclinic_wave_model(arch;
    Nλ, Nφ, Nz, H=30e3, Δt=0.8, halo=(H, H, 4))

output_dir = joinpath(get(ENV, "SCRATCH", pwd()), "model_dumps", "test_jld2_$(Ndev)gpu")

xy_fields = [:ρ, :ρu, :ρv, :ρw, :ρθ]
slices = [(f, :xy, [1, 32, 64]) for f in xy_fields]

# ── Save BEFORE compile ──────────────────────────────────────────────

dump1 = joinpath(output_dir, "before_compile")
rank == 0 && @info "[$rank] === Saving BEFORE compile ===" now(UTC)
GordonBell25.save_model_state(dump1, model, arch; label="output", slices=slices)
rank == 0 && @info "[$rank] Saved before compile" now(UTC)

# ── Compile ───────────────────────────────────────────────────────────

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all)

@info "[$rank] Compiling first_time_step!..." now(UTC)
rfirst! = @time @compile compile_options=compile_options first_time_step!(model)

# ── Save AFTER compile, BEFORE execution ──────────────────────────────

dump2 = joinpath(output_dir, "after_compile")
rank == 0 && @info "[$rank] === Saving AFTER compile, BEFORE execution ===" now(UTC)
GordonBell25.save_model_state(dump2, model, arch; label="output", slices=slices)
rank == 0 && @info "[$rank] Saved after compile" now(UTC)

# ── Execute one step ──────────────────────────────────────────────────

@info "[$rank] Running first_time_step!..." now(UTC)
@time "[$rank] first_time_step!" rfirst!(model)

# ── Save AFTER execution ──────────────────────────────────────────────

dump3 = joinpath(output_dir, "after_execution")
rank == 0 && @info "[$rank] === Saving AFTER execution ===" now(UTC)
GordonBell25.save_model_state(dump3, model, arch; label="output", slices=slices)
rank == 0 && @info "[$rank] Saved after execution" now(UTC)

@info "[$rank] Done!" now(UTC)
