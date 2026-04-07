using Dates
@info "Starting baroclinic instability simulation with output" now(UTC)

using ArgParse

const args_settings = ArgParseSettings()
@add_arg_table! args_settings begin
    "--grid-x"
        help = "Per-device tensor size in x (interior + 2H per device, scaled by Rx)."
        default = 256
        arg_type = Int
    "--grid-y"
        help = "Per-device tensor size in y (interior + 2H per device, scaled by Ry)."
        default = 128
        arg_type = Int
    "--grid-z"
        help = "Number of grid points on the z axis."
        default = 16
        arg_type = Int
    "--precision"
        help = "Number of bits of precision"
        default = 64
        arg_type = Int
    "--n-inner"
        help = "Time steps per compiled inner loop call."
        default = 128
        arg_type = Int
    "--n-outer"
        help = "Number of outer loop iterations (each runs n-inner steps)."
        default = 16
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
using JLD2

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

# --- Model setup ---
# Tx = grid-x * Rx is the total tensor dimension in x, guaranteed divisible by Rx.
# Nx = Tx - 2H is the interior (H = default halo size for the physics).

H = 8
Tx = parsed_args["grid-x"] * Rx
Ty = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nx = Tx - 2H
Ny = Ty - 2H

# Dynamic Δt from CFL at the highest latitude (80°).
# Split-explicit free surface handles fast barotropic waves;
# the baroclinic step is limited by internal wave speed (~3 m/s).
CFL_target  = 0.5
c_max       = 3.0        # m/s, internal gravity wave speed (conservative)
R_earth     = 6371220.0  # m
φ_max       = 80.0       # degrees, grid spans (-80, 80)
Δx_min      = (360.0 / Nx) * (π / 180) * R_earth * cosd(φ_max)
Δt          = CFL_target * Δx_min / c_max

@info "[$rank] Generating model (Nx=$Nx, Ny=$Ny, Nz=$Nz, Δt=$(round(Δt; sigdigits=3))s)..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt)
@info "[$rank] allocations" GordonBell25.allocatorstats()
@show model

# --- Compile ---

Ninner = parsed_args["n-inner"]
Nouter = parsed_args["n-outer"]

if local_arch isa Oceananigans.ReactantState
    Ninner = if Ndev == 1
        ConcreteRNumber(Ninner)
    else
        sharding = Sharding.NamedSharding(arch.connectivity, ())
   	    ConcreteRNumber(Ninner; sharding)
    end
end

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=["enzymexla.kernel_call", "(::Reactant.Compiler.LLVMFunc", "ka_with_reactant", "(::KernelAbstractions.Kernel", "var\"#_launch!;_launch!"])

@info "[$rank] Compiling first_time_step!..." now(UTC)
rfirst! = if local_arch isa Oceananigans.ReactantState
     @time "[$rank] compile first_time_step!" @compile compile_options=compile_options first_time_step!(model)
else
     first_time_step!
end

@info "[$rank] Compiling loop!..." now(UTC)
compiled_loop! = if local_arch isa Oceananigans.ReactantState
     @time "[$rank] compile loop!" @compile compile_options=compile_options loop!(model, Ninner)
else
     loop!
end

@info "[$rank] allocations" GordonBell25.allocatorstats()

# --- Output setup ---

output_dir = joinpath(@__DIR__, "output", jobid_procid)
rank == 0 && mkpath(output_dir)

"""
    save_snapshot!(model, output_dir, step; rank)

Save surface fields to JLD2. `Array()` on sharded ConcreteRArrays is a collective
all-gather (all processes must call it), so every rank participates in the transfer
but only rank 0 writes to disk.
"""
function save_snapshot!(model, output_dir, step; rank, Nx, Ny, Nz)
    # All ranks must call Array() — it is a collective operation
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    T = Array(interior(model.tracers.T))
    S = Array(interior(model.tracers.S))
    η = Array(interior(model.free_surface.η))

    if rank == 0
        path = joinpath(output_dir, @sprintf("snapshot_%04d.jld2", step))
        jldsave(path; u, v, T, S, η,
            metadata = Dict(
                "step" => step,
                "Nx" => Nx, "Ny" => Ny, "Nz" => Nz,
                "Ndev" => Ndev, "Rx" => Rx, "Ry" => Ry,
            ),
        )
        @info "[$rank] Saved snapshot $step → $path" now(UTC)
    end
end

# --- Run ---

@info "[$rank] Running first_time_step!..." now(UTC)
@time "[$rank] first time step" rfirst!(model)
@info "[$rank] allocations" GordonBell25.allocatorstats()

# Warmup loop (JIT / XLA caching)
@info "[$rank] Warmup loop..." now(UTC)
@time "[$rank] warmup loop" compiled_loop!(model, Ninner)
@info "[$rank] allocations" GordonBell25.allocatorstats()

# Save initial state
@info "[$rank] Saving initial snapshot..." now(UTC)
@time "[$rank] save snapshot 0" save_snapshot!(model, output_dir, 0; rank, Nx, Ny, Nz)

# Outer simulation loop
for i in 1:Nouter
    @info "[$rank] Outer iteration $i/$Nouter..." now(UTC)
    @time "[$rank] loop iteration $i" compiled_loop!(model, Ninner)
    @time "[$rank] save snapshot $i" save_snapshot!(model, output_dir, i; rank, Nx, Ny, Nz)
    @info "[$rank] allocations" GordonBell25.allocatorstats()
end

@info "[$rank] Done! Total steps: $(1 + (1 + Nouter) * parsed_args["n-inner"])" now(UTC)
