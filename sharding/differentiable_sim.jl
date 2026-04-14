#!/usr/bin/env julia
#
# Differentiable atmosphere simulation — single node, Reactant + Enzyme.
#
# Builds a small moist baroclinic wave model (analytic IC), defines a scalar
# loss (mean T² after N time steps), and compiles the backward pass via
# Enzyme reverse mode. The gradient dmodel mirrors the model structure:
# each prognostic field in dmodel holds ∂L/∂(field₀).
#
# Usage (inside uenv, single node):
#   julia --project=.. differentiable_sim.jl

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
using Enzyme
using Serialization
using Printf
using CUDA

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.preamble()
GordonBell25.initialize(; single_gpu_per_process=false)

local_arch = ReactantState()
arch = local_arch

Ndev = length(Reactant.devices())
@show Ndev

Rx, Ry = factors(Ndev)
if Ndev > 1
    arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
end

rank = Ndev == 1 ? 0 : Reactant.Distributed.local_rank()

# ─── Grid & model (small, analytic IC) ───────────────────────────────

Nλ, Nφ, Nz = 32, 32, 8
H = 30e3
Δt = 0.01
halo = (4, 4, 4)

@info "[$rank] Building model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz)" now(UTC)
model = GordonBell25.moist_baroclinic_wave_model(arch;
    Nλ, Nφ, Nz, H, Δt, halo,
    with_microphysics = false,
    with_surface_fluxes = false,
    initial_conditions_path = nothing,
    interpolation_type = :nearest)

@info "[$rank] Model built" now(UTC)

FT = eltype(model.grid)

# ─── Loss: mean T² after nsteps ──────────────────────────────────────
#
# No reset — the model's current state IS the input.  time_step!
# mutates in place; @trace with checkpointing lets Enzyme reverse
# through the loop.

function loss(model, Δt, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

# ─── Enzyme backward pass wrapper ────────────────────────────────────
#
# dmodel is the "shadow" — same structure as model, zero-initialized.
# After autodiff, every mutable array in dmodel holds the gradient of
# the loss w.r.t. that array's values at the start of loss().

function grad_loss(model, dmodel, Δt, nsteps)
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return loss_value
end

# ─── Compile ──────────────────────────────────────────────────────────

nsteps = 1

@info "[$rank] Creating shadow model (Enzyme.make_zero)..." now(UTC)
dmodel = Enzyme.make_zero(model)

@info "[$rank] Compiling backward pass (nsteps=$nsteps)..." now(UTC)
compiled_grad = @time "[$rank] compile grad_loss" Reactant.@compile(
    raise=true, raise_first=true, sync=true,
    grad_loss(model, dmodel, FT(Δt), nsteps)
)

@info "[$rank] Running backward pass..." now(UTC)
loss_val = @time "[$rank] grad_loss execution" compiled_grad(
    model, dmodel, FT(Δt), nsteps)

@info "[$rank] Loss value = $loss_val"

# ─── Extract gradients to host ────────────────────────────────────────

grad_fields = [
    (:dρ,   dynamics_density(dmodel.dynamics)),
    (:dρu,  dmodel.momentum.ρu),
    (:dρv,  dmodel.momentum.ρv),
    (:dρw,  dmodel.momentum.ρw),
    (:dρθ,  dmodel.formulation.potential_temperature_density),
    (:dρqᵛ, dmodel.moisture_density),
]

grad_data = Dict{Symbol, Array}()
for (name, field) in grad_fields
    arr = Array(interior(field))
    grad_data[name] = arr
    nz = count(==(0), arr)
    nn = count(isnan, arr)
    @info @sprintf("  %-6s  size=%-18s  extrema=(%+.4e, %+.4e)  zeros=%d  NaN=%d",
                   name, string(size(arr)), extrema(arr)..., nz, nn)
end

# ─── Save to disk ─────────────────────────────────────────────────────

out_dir = joinpath(@__DIR__, "output", "differentiable_sim")
mkpath(out_dir)

out_path = joinpath(out_dir, "grad_output.dat")
open(out_path, "w") do io
    Serialization.serialize(io, Dict(
        :loss_value => loss_val,
        :nsteps     => nsteps,
        :Δt         => Δt,
        :Nλ         => Nλ,
        :Nφ         => Nφ,
        :Nz         => Nz,
        :gradients  => grad_data,
    ))
end

@info "[$rank] Saved to $out_path" now(UTC)
@info "[$rank] Done!" now(UTC)
