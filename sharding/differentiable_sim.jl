#!/usr/bin/env julia
#
# Differentiable atmosphere simulation — Reactant + Enzyme.
#
# Loads a 1° (360×160×64) atmosphere checkpoint, interpolates onto the
# target grid, defines a scalar loss (mean T² after N time steps), and
# compiles the backward pass via Enzyme reverse mode.
#
# Grid size is set via --grid-x / --grid-y / --grid-z (per-GPU points
# including halo), matching the convention in the sharded simulation
# scripts and alps_scaling_test.jl.
#
# Usage:
#   julia --project=.. differentiable_sim.jl                        # 128×128×64 per GPU
#   julia --project=.. differentiable_sim.jl --grid-x 64 --grid-y 64 --grid-z 32

using Dates
@info "Differentiable simulation starting" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: factors, is_distributed_env_present
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics
using Breeze.AtmosphereModels: dynamics_density
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior
using Statistics: mean
using Reactant
using Reactant: @trace
using ReactantCore: Periodic as CheckpointPeriodic
using Reactant.Enzyme
using Serialization
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
#
# The compiled Thunk's result copy-back calls `Base.copy` on IFRT
# buffers for dealiasing.  The default `ifrt_copy_array` fails when
# the buffer spans non-addressable devices.  We override it to
# disassemble → copy per-shard → reassemble, which only touches
# addressable devices.

let IFRT = Reactant.XLA.IFRT, MLIR = Reactant.MLIR
    function Base.copy(b::IFRT.Array)
        try
            return IFRT.Array(GC.@preserve b begin
                MLIR.API.ifrt_copy_array(b.buffer)
            end)
        catch e
            occursin("non-addressable device", string(e)) || rethrow(e)

            sharding  = Reactant.XLA.sharding(b)   # IFRT.Sharding (prevents GC of b)
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
                    2,  # kDonateInput
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

# ─── Grid (per-GPU × partition, same convention as sharded scripts) ──

H_halo = 4
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]
Nλ = Tλ - 2H_halo
Nφ = Tφ - 2H_halo

H = 30e3
Δt = 0.03
halo = (H_halo, H_halo, 4)

ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                   "atmosphere_no_microphysics_1deg_14day.jld2")
isfile(ic_path) || error("IC file not found: $ic_path — run simulations/download_atmosphere_ic_artifact.jl first")

@info "[$rank] Grid: per-GPU=$(parsed_args["grid-x"])×$(parsed_args["grid-y"])  partition=$(Rx)×$(Ry)  total interior=$(Nλ)×$(Nφ)×$(Nz)"
@info "[$rank] Building model with real IC" now(UTC)
model = @time "[$rank] model build" GordonBell25.moist_baroclinic_wave_model(arch;
    Nλ, Nφ, Nz, H, Δt, halo,
    with_microphysics = false,
    with_surface_fluxes = false,
    initial_conditions_path = ic_path,
    interpolation_type = :linear)

@info "[$rank] Model built" now(UTC)
@info "[$rank] allocations (after model build)" alloc()
@show model

FT = eltype(model.grid)

ngpu_tag = "ngpu=$(lpad(Ndev, 5, '0'))"
out_dir = joinpath(@__DIR__, "output", "differentiable_sim", ngpu_tag)

# NOTE: initial fields save moved to after backward pass to avoid
# local_shards_to_host touching live model arrays before compile/execute.

# ─── Loss: Var(max(w, 0)) after nsteps ───────────────────────────────
#
# Variance of the positive vertical velocity.
# model.velocities.w is the diagnosed w = ρw / ρ, already consistent.
# Var(x) = E[x²] − E[x]².
#
# Ninner_val is a Julia literal baked into the IR so that the @trace
# while-loop has fully static shapes (required by Shardy propagation).

const Ninner_val = 100

function loss(model, Δt)
    @trace mincut=true checkpointing=CheckpointPeriodic(Ninner_val) track_numbers=false for _ in 1:Ninner_val
        time_step!(model, Δt)
    end
    w     = interior(model.velocities.w)
    w_pos = max.(w, zero(FT))
    μ     = mean(w_pos)
    return mean(w_pos .* w_pos) - μ * μ
end

# ─── Enzyme backward pass wrapper ────────────────────────────────────
#
# dmodel is the "shadow" — same structure as model, zero-initialized.
# After autodiff, every mutable array in dmodel holds the gradient of
# the loss w.r.t. that array's values at the start of loss().

function grad_loss(model, dmodel, Δt)
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Const(Δt))
    return loss_value
end

# ─── Compile ──────────────────────────────────────────────────────────

@info "[$rank] Creating shadow model (Enzyme.make_zero)..." now(UTC)
dmodel = @time "[$rank] make_zero" Enzyme.make_zero(model)
@info "[$rank] allocations (after make_zero)" alloc()
@show dmodel

@info "[$rank] Compiling backward pass (Ninner_val=$Ninner_val, literal constant)..." now(UTC)
compiled_grad = @time "[$rank] compile grad_loss" Reactant.@compile(
    raise=true, raise_first=true, sync=true,
    grad_loss(model, dmodel, FT(Δt))
)
@info "[$rank] allocations (after compile)" alloc()

@info "[$rank] Running backward pass..." now(UTC)
loss_val = @time "[$rank] grad_loss execution" compiled_grad(
    model, dmodel, FT(Δt))

@info "[$rank] Loss value = $loss_val"
@info "[$rank] allocations (after execution)" alloc()

# ─── Save outputs via save_model_state ────────────────────────────────

z_levels = [:bottom, :middle, :top]
save_fields = [:ρ, :ρu, :ρv, :ρw, :ρθ, :ρqᵛ]
save_slices = [(f, :xy, z_levels) for f in save_fields]

@info "[$rank] Saving gradient fields..." now(UTC)
GordonBell25.save_model_state(joinpath(out_dir, "gradients"), dmodel, arch;
    label="output", slices=save_slices)

@info "[$rank] Saving initial fields..." now(UTC)
GordonBell25.save_model_state(joinpath(out_dir, "initial_fields"), model, arch;
    label="output", slices=save_slices)

@info "[$rank] allocations (final)" alloc()
@info "[$rank] Done!" now(UTC)
