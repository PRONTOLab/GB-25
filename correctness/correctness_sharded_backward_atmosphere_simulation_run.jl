#!/usr/bin/env julia
#
# CI correctness test: sharded Enzyme backward pass for the atmosphere model.
#
# Builds a tiny sharded atmosphere model from analytic IC (no file download),
# compiles and runs the Enzyme reverse-mode backward pass, then validates
# that the loss is finite and gradients are non-zero / non-NaN.
#
# Grid is deliberately tiny to fit within CI runner memory (~7 GB) since
# the backward pass requires a shadow model (2× memory) plus Enzyme
# compilation overhead.
#
# Naming follows CompileOrRun.yml convention:
#   correctness/correctness_sharded_backward_atmosphere_simulation_run.jl

using Dates
@info "Sharded backward-pass correctness test starting" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: factors, is_distributed_env_present
using Breeze
using Breeze.AtmosphereModels: dynamics_density
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior
using Statistics: mean
using Reactant
using Reactant: @trace
using Reactant.Enzyme
using CUDA

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 16,
    grid_y_default = 16,
    grid_z_default = 8,
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
                    2,  # kDonateInput
                )
            end
            return IFRT.Array(buf)
        end
    end
end

# ─── Architecture ─────────────────────────────────────────────────────

local_arch = ReactantState()
arch = local_arch

Ndev = length(Reactant.devices())
@show Ndev

Rx, Ry = factors(Ndev)
if Ndev > 1
    arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
end

rank = Ndev == 1 ? 0 : Reactant.Distributed.local_rank()

# ─── Tiny grid for CI ────────────────────────────────────────────────

H_halo = 4
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]
Nλ = Tλ - 2H_halo
Nφ = Tφ - 2H_halo

H = 30e3
Δt = 1e-3
halo = (H_halo, H_halo, 4)
FT = Oceananigans.defaults.FloatType

@info "[$rank] Grid: per-GPU=$(parsed_args["grid-x"])×$(parsed_args["grid-y"])  partition=$(Rx)×$(Ry)  total interior=$(Nλ)×$(Nφ)×$(Nz)"

# ─── Build model (analytic IC, no file) ──────────────────────────────

@info "[$rank] Building model with analytic IC..." now(UTC)
model = @time "[$rank] model build" GordonBell25.moist_baroclinic_wave_model(arch;
    Nλ, Nφ, Nz, H, Δt, halo,
    with_microphysics   = false,
    with_surface_fluxes = false,
    initial_conditions_path = nothing,
    interpolation_type = :linear)

@info "[$rank] Model built" now(UTC)
@show model

# ─── Compile update_state! and run it once ───────────────────────────

@info "[$rank] Compiling update_state!..." now(UTC)
compiled_update! = @time "[$rank] compile update_state!" Reactant.@compile(
    sync=true, raise=true,
    update_state!(model, compute_tendencies=false))

@info "[$rank] Running update_state!..." now(UTC)
@time "[$rank] update_state!" compiled_update!(model)

# ─── Loss function ───────────────────────────────────────────────────
#
# Var(max(w, 0)) after Ninner_val time steps.
# Identical to forward_backward_sim.jl but with fewer steps.

const Ninner_val = 2

function loss(model, Δt)
    @trace mincut=true checkpointing=false track_numbers=false for _ in 1:Ninner_val
        Oceananigans.TimeSteppers.time_step!(model, Δt)
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

# ─── Shadow model ────────────────────────────────────────────────────

@info "[$rank] Creating shadow model (Enzyme.make_zero)..." now(UTC)
dmodel = @time "[$rank] make_zero" Enzyme.make_zero(model)

# ─── Compile backward pass ───────────────────────────────────────────

@info "[$rank] Compiling backward pass (Ninner_val=$Ninner_val)..." now(UTC)
compiled_grad = @time "[$rank] compile grad_loss" Reactant.@compile(
    raise=true, raise_first=true, sync=true,
    grad_loss(model, dmodel, FT(Δt)))

# ─── Run backward pass ──────────────────────────────────────────────

@info "[$rank] Running backward pass..." now(UTC)
loss_val = @time "[$rank] grad_loss execution" compiled_grad(
    model, dmodel, FT(Δt))

@info "[$rank] Loss value = $loss_val"

# ─── Validate results ───────────────────────────────────────────────
#
# The backward pass must produce:
#   1. A finite (non-NaN, non-Inf) loss value
#   2. At least some non-zero gradients in the shadow model

passed = true

if isnan(loss_val) || isinf(loss_val)
    @error "FAIL: loss is not finite" loss_val
    passed = false
else
    @info "PASS: loss is finite" loss_val
end

grad_fields = [
    ("dρ",   dynamics_density(dmodel.dynamics)),
    ("dρu",  dmodel.momentum.ρu),
    ("dρv",  dmodel.momentum.ρv),
    ("dρw",  dmodel.momentum.ρw),
    ("dρθ",  dmodel.formulation.potential_temperature_density),
    ("dρqᵛ", dmodel.moisture_density),
]

any_nonzero = false
for (name, field) in grad_fields
    ifrt_arr = Reactant.ancestor(field)
    local_shards = Reactant.XLA.IFRT.disassemble_into_single_device_arrays(ifrt_arr.data, true)
    n_nan = 0
    n_nonzero = 0
    total = 0
    lo = Inf
    hi = -Inf
    for shard in local_shards
        shard_size = reverse(size(shard))
        buf = Base.Array{eltype(ifrt_arr)}(undef, shard_size...)
        Reactant.XLA.to_host(shard, buf, Reactant.Sharding.NoSharding())
        total += length(buf)
        for v in buf
            if isnan(v)
                n_nan += 1
            else
                lo = min(lo, v)
                hi = max(hi, v)
                if v != 0
                    n_nonzero += 1
                end
            end
        end
    end
    ex = total == n_nan ? (NaN, NaN) : (lo, hi)
    @info "[$rank] $name  size=$total  extrema=$ex  NaN=$n_nan  nonzero=$n_nonzero"

    if n_nan > 0
        @error "FAIL: $name contains NaN" n_nan total
        passed = false
    end
    if n_nonzero > 0
        any_nonzero = true
    end
end

if !any_nonzero
    @error "FAIL: all gradient fields are identically zero"
    passed = false
else
    @info "PASS: gradients contain non-zero values"
end

@info "[$rank] Backward pass correctness test $(passed ? "PASSED ✓" : "FAILED ✗")" now(UTC)

passed || error("Backward pass correctness test failed")
