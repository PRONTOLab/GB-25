#!/usr/bin/env julia
#
# Benchmark: forward pass vs backward pass (same number of time steps).
#
# Compiles both a forward-only loop and the Enzyme backward pass for
# Nsteps time steps, then times repeated executions of each.
#
# Usage:
#   julia --project=.. bench_fwd_bwd.jl                        # 128×128×64 per GPU
#   julia --project=.. bench_fwd_bwd.jl --grid-x 64 --grid-y 64 --grid-z 32

using Dates
@info "Forward / backward benchmark starting" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: factors, is_distributed_env_present, loop!
using Oceananigans.TimeSteppers: update_state!
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics
using Breeze.AtmosphereModels: dynamics_density
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior
using Statistics: mean, median
using Reactant
using Reactant: @trace
using ReactantCore: Periodic as CheckpointPeriodic
using Reactant.Enzyme
using Printf
using CUDA

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 128,
    grid_y_default = 128,
    grid_z_default = 64,
)

Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.preamble()
GordonBell25.initialize(; single_gpu_per_process=false)

# ─── Workaround for Reactant #2233: distributed IFRT copy ────────────

let IFRT = Reactant.XLA.IFRT, MLIR = Reactant.MLIR
    function Base.copy(b::IFRT.Array)
        try
            return IFRT.Array(GC.@preserve b begin
                MLIR.API.ifrt_copy_array(b.buffer)
            end)
        catch e
            occursin("non-addressable device", string(e)) || rethrow(e)

            sharding  = Reactant.XLA.sharding(b)
            client    = Reactant.XLA.client(b)
            sdas      = IFRT.disassemble_into_single_device_arrays(b, true)

            copied_ptrs = Vector{Ptr{Nothing}}(undef, length(sdas))
            for (i, sda) in enumerate(sdas)
                copied_ptrs[i] = GC.@preserve sda begin
                    MLIR.API.ifrt_copy_array(sda.buffer)
                end
            end

            arr_shape = collect(Int64, reverse(size(b)))
            buf = GC.@preserve b client sharding begin
                MLIR.API.ifrt_client_assemble_array_from_single_shards(
                    client.client,
                    length(arr_shape), arr_shape,
                    sharding.ptr,
                    length(copied_ptrs), copied_ptrs,
                    2,
                )
            end
            return IFRT.Array(buf)
        end
    end
end

local_arch = ReactantState()
arch = local_arch

Ndev = length(Reactant.devices())
@show Ndev

Rx, Ry = factors(Ndev)
if Ndev > 1
    arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
end

rank = Ndev == 1 ? 0 : Reactant.Distributed.local_rank()

alloc() = GordonBell25.allocatorstats()

@info "[$rank] allocations (after init)" alloc()

# ─── Grid ─────────────────────────────────────────────────────────────

H_halo = 4
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]
Nλ = Tλ - 2H_halo
Nφ = Tφ - 2H_halo

H = 30e3
Δt = 0.01
halo = (H_halo, H_halo, 4)

ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                   "atmosphere_no_microphysics_1deg_14day.jld2")
isfile(ic_path) || error("IC file not found: $ic_path")

@info "[$rank] Grid: per-GPU=$(parsed_args["grid-x"])×$(parsed_args["grid-y"])  partition=$(Rx)×$(Ry)  total interior=$(Nλ)×$(Nφ)×$(Nz)"
@info "[$rank] Building model with real IC" now(UTC)
model = @time "[$rank] model build" GordonBell25.moist_baroclinic_wave_model(arch;
    Nλ, Nφ, Nz, H, Δt, halo,
    with_microphysics = true,
    with_surface_fluxes = false,
    initial_conditions_path = ic_path,
    interpolation_type = :linear)

FT = eltype(model.grid)

# ─── Benchmark parameters ────────────────────────────────────────────

const Nsteps  = 4      # time steps per invocation (must match for fwd & bwd)
const Ntrials = 10     # timed repetitions (after warmup)

@info "[$rank] Benchmark config" Nsteps Ntrials Ndev

# ═══════════════════════════════════════════════════════════════════════
# FORWARD PASS
# ═══════════════════════════════════════════════════════════════════════

Nfwd = if Ndev > 1
    fwd_sharding = Sharding.NamedSharding(arch.connectivity, ())
    ConcreteRNumber(Nsteps; sharding=fwd_sharding)
else
    ConcreteRNumber(Nsteps)
end

compile_options = Reactant.CompileOptions(; sync=true, raise=true,
    strip_llvm_debuginfo=true, strip=:all)

@info "[$rank] Compiling update_state!..." now(UTC)
compiled_update! = @time "[$rank] compile update_state!" Reactant.@compile(
    compile_options=compile_options,
    update_state!(model, compute_tendencies=false))

@info "[$rank] Compiling forward loop! (Nsteps=$Nsteps)..." now(UTC)
compiled_fwd! = @time "[$rank] compile forward" Reactant.@compile(
    compile_options=compile_options,
    loop!(model, Nfwd))

@info "[$rank] Forward compile done" now(UTC) alloc()

# ═══════════════════════════════════════════════════════════════════════
# BACKWARD PASS
# ═══════════════════════════════════════════════════════════════════════

const Ninner_val = Nsteps

function loss(model, Δt)
    @trace mincut=true checkpointing=false track_numbers=false for _ in 1:Ninner_val
        time_step!(model, Δt)
    end
    w     = interior(model.velocities.w)
    w_pos = max.(w, zero(FT))
    μ     = mean(w_pos)
    return mean(w_pos .* w_pos) - μ * μ
end

function grad_loss(model, dmodel, Δt)
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Const(Δt))
    return loss_value
end

@info "[$rank] Creating shadow model (Enzyme.make_zero)..." now(UTC)
dmodel = @time "[$rank] make_zero" Enzyme.make_zero(model)

@info "[$rank] Compiling backward pass (Ninner_val=$Ninner_val)..." now(UTC)
compiled_bwd = @time "[$rank] compile backward" Reactant.@compile(
    raise=true, raise_first=true, sync=true,
    grad_loss(model, dmodel, FT(Δt))
)

@info "[$rank] Backward compile done" now(UTC) alloc()

# ═══════════════════════════════════════════════════════════════════════
# WARMUP
# ═══════════════════════════════════════════════════════════════════════

@info "[$rank] Warmup: update_state!..." now(UTC)
compiled_update!(model)

@info "[$rank] Warmup: forward..." now(UTC)
compiled_fwd!(model, Nfwd)

@info "[$rank] Warmup: backward..." now(UTC)
compiled_bwd(model, dmodel, FT(Δt))

@info "[$rank] Warmup done" now(UTC)

# ═══════════════════════════════════════════════════════════════════════
# TIMED RUNS
# ═══════════════════════════════════════════════════════════════════════

fwd_times = Float64[]
bwd_times = Float64[]

@info "[$rank] Running $Ntrials forward trials..." now(UTC)
for i in 1:Ntrials
    t = @elapsed compiled_fwd!(model, Nfwd)
    push!(fwd_times, t)
end

@info "[$rank] Running $Ntrials backward trials..." now(UTC)
for i in 1:Ntrials
    t = @elapsed compiled_bwd(model, dmodel, FT(Δt))
    push!(bwd_times, t)
end

# ═══════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════

fmt(v) = @sprintf("%.4f", v)

fwd_med = median(fwd_times)
bwd_med = median(bwd_times)
ratio   = bwd_med / fwd_med

@info """
[$rank] ══════════════════════════════════════════════════════
  Benchmark results  (Nsteps=$Nsteps, Ntrials=$Ntrials, Ndev=$Ndev)
  Grid: $(parsed_args["grid-x"])×$(parsed_args["grid-y"])×$(parsed_args["grid-z"]) per GPU

  FORWARD  (loop! with $Nsteps steps):
    min    = $(fmt(minimum(fwd_times))) s
    median = $(fmt(fwd_med)) s
    max    = $(fmt(maximum(fwd_times))) s
    all    = $(join(map(fmt, fwd_times), ", ")) s

  BACKWARD (Enzyme reverse with $Nsteps steps):
    min    = $(fmt(minimum(bwd_times))) s
    median = $(fmt(bwd_med)) s
    max    = $(fmt(maximum(bwd_times))) s
    all    = $(join(map(fmt, bwd_times), ", ")) s

  RATIO  backward / forward = $(fmt(ratio))×
══════════════════════════════════════════════════════════════
""" now(UTC)

@info "[$rank] allocations (final)" alloc()
@info "[$rank] Done!" now(UTC)
