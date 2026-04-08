using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors, is_distributed_env_present
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 8,
)

Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)

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

H = 8
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nλ = Tλ - 2H
Nφ = Tφ - 2H

CFL_target = 0.5
c_sound = 330.0        # m/s
R_earth = 6371220.0    # m
φ_cap = 85.0           # degrees; grid latitude extent (−85°, 85°)
column_height = 30e3   # m; default column height in moist_baroclinic_wave_model

Δz = column_height / Nz
Δλ_rad = 2π / Nλ
Δφ_rad = (2 * φ_cap) * (π / 180) / Nφ
Δx_zonal = R_earth * cosd(φ_cap) * Δλ_rad
Δy_merid = R_earth * Δφ_rad
Δ_min = min(Δz, Δx_zonal, Δy_merid)
Δt = CFL_target * Δ_min / c_sound

@info "[$rank] Generating atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, Δt=$(round(Δt; sigdigits=3))s)..." now(UTC)
model = GordonBell25.moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz, H=column_height, Δt, halo=(H, H, H))
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

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=["enzymexla.kernel_call", "(::Reactant.Compiler.LLVMFunc", "ka_with_reactant", "(::KernelAbstractions.Kernel", "var\"#_launch!;_launch!"])

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)

@info "[$rank] Compiling first_time_step!..." now(UTC)
mkpath(joinpath(profile_dir, "compile_first_time_step"))
rfirst! = Reactant.with_profiler(joinpath(profile_dir, "compile_first_time_step")) do
    if local_arch isa Oceananigans.ReactantState
         @time "[$rank] compile first_time_step!" @compile compile_options=compile_options first_time_step!(model)
    else
         first_time_step!
    end
end

@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] Compiling loop..." now(UTC)

mkpath(joinpath(profile_dir, "compile_loop"))
compiled_loop! = Reactant.with_profiler(joinpath(profile_dir, "compile_loop")) do
    if local_arch isa Oceananigans.ReactantState
         @time "[$rank] compile loop!" @compile compile_options=compile_options loop!(model, Ninner)
    else
         loop!
    end
end

@info "[$rank] allocations" GordonBell25.allocatorstats()

mkpath(joinpath(profile_dir, "first_time_step"))
@info "[$rank] Running first_time_step!..." now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "first_time_step")) do
    Reactant.Profiler.annotate("first_time_step"; metadata=Dict("step_num" => 0, "_r" => 1)) do
        @time "[$rank] first_time_step!" rfirst!(model)
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

checkpoint_dir = joinpath(@__DIR__, "checkpoints", jobid_procid)
@info "[$rank] Saving sharded checkpoint..." now(UTC)
@time "[$rank] checkpoint save" begin
    filepath = GordonBell25.save_model_state(checkpoint_dir, model, arch; label="final")
    @info "[$rank] Checkpoint saved to $filepath"
end

checkpoint_data_dir = joinpath(checkpoint_dir, "final")
@info "[$rank] Visualizing checkpoint..." now(UTC)
@time "[$rank] checkpoint visualize" begin
    GordonBell25.visualize_checkpoint(checkpoint_data_dir; halo=H)
end

@info "[$rank] Done!" now(UTC)
