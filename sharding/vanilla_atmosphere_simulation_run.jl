using Dates
@info "Vanilla (GPU/MPI) atmosphere simulation" now(UTC)

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors
using Breeze.AtmosphereModels: dynamics_density
using Oceananigans.TimeSteppers: update_state!
using Oceananigans
using Oceananigans.Units
using Printf
using MPI
using CUDA

MPI.Init()

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 8,
)

Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)

comm = MPI.COMM_WORLD
Ndev = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

nvis = length(CUDA.devices())
if nvis == 1
    # --gpu-bind=per_task:1 already restricts each task to one GPU (device 0)
elseif nvis > 1
    local_id = rank % nvis
    CUDA.device!(local_id)
end
@info "[$rank/$Ndev] GPU=$(CUDA.device()) visible=$nvis on $(gethostname())"

local_arch = Oceananigans.GPU()
arch = local_arch

if Ndev > 1
    Rx, Ry = factors(Ndev)
    arch = Oceananigans.Distributed(local_arch; partition = Oceananigans.Partition(Rx, Ry, 1))
    @info "[$rank/$Ndev] Distributed partition: Rx=$Rx, Ry=$Ry"
else
    Rx, Ry = 1, 1
end

H = 4
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nλ = Tλ - 2H
Nφ = Tφ - 2H

column_height = 30e3
Δt = 0.2

_ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                    "quarter_deg_day1_cloud_tau30.jld2")
initial_conditions_path = isfile(_ic_path) ? _ic_path : nothing

if initial_conditions_path !== nothing
    @info "[$rank] Initializing from file" initial_conditions_path
else
    @warn "[$rank] IC file not found at $_ic_path — using analytic IC"
end

@info "[$rank] Generating atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, Δt=$(round(Δt; sigdigits=3))s)..." now(UTC)
model = GordonBell25.moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz, H=column_height, Δt,
                                                 halo=(H, H, 4),
                                                 with_microphysics=false,
                                                 initial_conditions_path=initial_conditions_path,
                                                 interpolation_type=:linear)

@show model

# ─── NaN check utility ───────────────────────────────────────────────

function vanilla_nan_check(rank, label, model)
    @info "[$rank] NaN check: $label" now(UTC)
    for (name, field) in [
        ("ρ",   dynamics_density(model.dynamics)),
        ("ρu",  model.momentum.ρu),
        ("ρv",  model.momentum.ρv),
        ("ρw",  model.momentum.ρw),
        ("ρθ",  model.formulation.potential_temperature_density),
        ("ρqᵛ", model.moisture_density),
    ]
        data = Array(parent(field))
        n_nan = count(isnan, data)
        ex = n_nan == length(data) ? (NaN, NaN) : extrema(filter(!isnan, data))
        @info "[$rank]   $name  size=$(length(data))  extrema=$ex  NaN=$n_nan/$(length(data))"
    end
end

# ─── Post-construction field extrema ─────────────────────────────────

@info "[$rank] Post-construction field extrema (interior only):"
for (name, field) in pairs(Oceananigans.fields(model))
    data = Array(Oceananigans.interior(field))
    @info "[$rank]   $name: extrema = $(extrema(data)), any_nan = $(any(isnan, data))"
end

# ─── Time stepping ───────────────────────────────────────────────────

Ninner = 64

@info "[$rank] Running first_time_step!..." now(UTC)
@time "[$rank] first_time_step!" first_time_step!(model)

@info "[$rank] Running first loop ($Ninner steps)..." now(UTC)
@time "[$rank] first loop" begin
    for _ in 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
end

vanilla_nan_check(rank, "after first loop", model)

# ─── Save after first loop ───────────────────────────────────────────

jobid = get(ENV, "SLURM_JOB_ID", "local")
output_base = joinpath(@__DIR__, "output", "vanilla_$jobid")
mkpath(output_base)

_xy_fields = [:u, :v, :w, :θ, :qᵛ]
_xy_levels = [1, 2, 4, 8, 16]
_first_loop_slices = [(f, :xy, _xy_levels) for f in _xy_fields]
@time "[$rank] save after first loop" begin
    GordonBell25.save_model_state(joinpath(output_base, "after_first_loop"), model, arch;
        label = "output",
        slices = _first_loop_slices)
end

# ─── Second loop ─────────────────────────────────────────────────────

@info "[$rank] Running second loop ($Ninner steps)..." now(UTC)
@time "[$rank] second loop" begin
    for _ in 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
end

vanilla_nan_check(rank, "after second loop", model)

# ─── Output specification ────────────────────────────────────────────

xy_fields = [:u, :v, :w, :θ, :qᵛ]
xy_levels = [1, 2, 4, 8, 16]
output_slices = [(f, :xy, xy_levels) for f in xy_fields]

output_dir = joinpath(output_base, "blocks")
mkpath(output_dir)

# ─── Outer loop ──────────────────────────────────────────────────────

const Nouter = 1200
Ninner_val = 256

@info "[$rank] Starting outer loop: $Nouter blocks × $Ninner_val inner steps (Δt=$Δt)" now(UTC)

wall_start = time_ns()
for k in 1:Nouter
    t0 = time_ns()

    for _ in 1:Ninner_val
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end

    wall_block = (time_ns() - t0) / 1e9
    sim_time   = Ninner_val * k * Δt
    total_wall = (time_ns() - wall_start) / 1e9
    sypd       = (Ninner_val * Δt) / (365.25 * 86400 * wall_block) * 365.25

    @info @sprintf("[%d] block %d/%d wall=%.1fs sim=%.1fs SYPD=%.5f total_wall=%.0fs",
                    rank, k, Nouter, wall_block, sim_time, sypd, total_wall)

    block_dir = joinpath(output_dir, @sprintf("block_%04d", k))
    @time "[$rank] save block $k" begin
        GordonBell25.save_model_state(block_dir, model, arch;
            label = "output",
            slices = output_slices)
    end
    @info "[$rank] saved block $k" block_dir

    vanilla_nan_check(rank, "block $k", model)

    flush(stderr); flush(stdout)
end

@info "[$rank] Done!" now(UTC)
