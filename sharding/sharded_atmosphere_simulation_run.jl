using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors, is_distributed_env_present
using Breeze.AtmosphereModels: dynamics_density
using Oceananigans.TimeSteppers: update_state!
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

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = false
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

H = 4
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nλ = Tλ - 2H
Nφ = Tφ - 2H

column_height = 30e3   # m; default column height in moist_baroclinic_wave_model

# Vertical acoustic CFL is the binding constraint for ExplicitTimeStepping:
# Δt < Δz / c_s ≈ (30 km / 64) / 340 m/s ≈ 1.38 s, independent of horizontal
# resolution. Hardcode Δt below the limit and don't auto-derive.
Δt = 0.001

# File-based initialization. The artifact is downloaded by
# `simulations/download_atmosphere_ic_artifact.jl` and lives in the
# sibling `simulations/initial_conditions/` directory. Under sharding the
# loader builds the source field on the *single-rank* child architecture
# and `interpolate!` scatters into the sharded target.
# Falls back to analytic IC if the file is missing.

_ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                    "cascade_checkpoint.jld2")
# _ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
#                     "quarter_deg_day1_cloud_tau30.jld2")
# _ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
#                     "atmosphere_coarsened_1536x768x64.jld2")
# _ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
#                     "atmosphere_no_microphysics_1deg_14day.jld2")
initial_conditions_path = isfile(_ic_path) ? _ic_path : nothing

if initial_conditions_path !== nothing
    @info "[$rank] Initializing from file" initial_conditions_path
else
    @warn "[$rank] IC file not found at $_ic_path — using analytic IC"
end
# initial_conditions_path = nothing

@info "[$rank] Generating atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, Δt=$(round(Δt; sigdigits=3))s)..." now(UTC)
model = GordonBell25.moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz, H=column_height, Δt,
                                                 halo=(H, H, 4),
                                                 with_microphysics=true,
                                                 cloud_formation_τ_relax=120.0,
                                                 initial_conditions_path=initial_conditions_path,
                                                 sst_anomaly = 2,
                                                 interpolation_type=:linear)
@info "[$rank] allocations" GordonBell25.allocatorstats()

@show model

Ninner = 64

if local_arch isa Oceananigans.ReactantState
    Ninner = if Ndev == 1
        ConcreteRNumber(Ninner)
    else
        sharding = Sharding.NamedSharding(arch.connectivity, ())
   	    ConcreteRNumber(Ninner; sharding)
    end
end

function make_dt(val, local_arch, arch, Ndev)
    if local_arch isa Oceananigans.ReactantState
        if Ndev == 1
            ConcreteRNumber(Float32(val))
        else
            sharding = Sharding.NamedSharding(arch.connectivity, ())
            ConcreteRNumber(Float32(val); sharding)
        end
    else
        Float32(val)
    end
end

Δt_r = make_dt(Δt, local_arch, arch, Ndev)

function loop_with_dt!(model, Ninner, Δt)
    Reactant.Profiler.annotate("loop") do
        @trace track_numbers=false for _ = 1:Ninner
            Oceananigans.TimeSteppers.time_step!(model, Δt)
        end
    end
    return nothing
end

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all)

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)

@info "[$rank] Compiling first_time_step!..." now(UTC)
mkpath(joinpath(profile_dir, "compile_first_time_step"))
rfirst! = begin
    if local_arch isa Oceananigans.ReactantState
         @time "[$rank] compile first_time_step!" @compile compile_options=compile_options update_state!(model, compute_tendencies=false)
    else
         first_time_step!
    end
end

@info "[$rank] allocations" GordonBell25.allocatorstats()
@info "[$rank] Compiling loop (Ninner=64)..." now(UTC)

mkpath(joinpath(profile_dir, "compile_loop"))
compiled_loop! = begin
    if local_arch isa Oceananigans.ReactantState
         @time "[$rank] compile loop!" @compile compile_options=compile_options loop_with_dt!(model, Ninner, Δt_r)
    else
         (model, N, dt) -> loop!(model, N)
    end
end

@info "[$rank] allocations" GordonBell25.allocatorstats()

# ─── first_time_step! (update_state!) ─────────────────────────────────
mkpath(joinpath(profile_dir, "first_time_step"))
@info "[$rank] Running first_time_step!..." now(UTC)
@time "[$rank] first_time_step!" rfirst!(model)
@info "[$rank] allocations" GordonBell25.allocatorstats()

# ─── NaN check helper ─────────────────────────────────────────────────
function local_nan_check(rank, label, model)
    @info "[$rank] NaN check: $label" now(UTC)
    for (name, field) in [
        ("ρ",   dynamics_density(model.dynamics)),
        ("ρu",  model.momentum.ρu),
        ("ρv",  model.momentum.ρv),
        ("ρw",  model.momentum.ρw),
        ("ρθ",  model.formulation.potential_temperature_density),
        ("ρqᵛ", model.moisture_density),
    ]
        ifrt_arr = Reactant.ancestor(field)
        local_shards = Reactant.XLA.IFRT.disassemble_into_single_device_arrays(ifrt_arr.data, true)
        total_nan = 0
        total_len = 0
        lo = Inf
        hi = -Inf
        for shard in local_shards
            shard_size = reverse(size(shard))
            buf = Base.Array{eltype(ifrt_arr)}(undef, shard_size...)
            Reactant.XLA.to_host(shard, buf, Reactant.Sharding.NoSharding())
            total_len += length(buf)
            for v in buf
                if isnan(v)
                    total_nan += 1
                else
                    lo = min(lo, v)
                    hi = max(hi, v)
                end
            end
        end
        ex = total_len == total_nan ? (NaN, NaN) : (lo, hi)
        @info "[$rank] $name  local_size=$total_len  extrema=$ex  NaN=$total_nan/$total_len"
    end
end

# ─── Output specification ─────────────────────────────────────────────
xy_fields = [:u, :v, :w, :θ, :qᵛ, :qᶜˡ, :qᶜⁱ]
xy_levels = [1, 2, 4, 8, 16]

yz_fields = [:w, :qᶜˡ, :qᶜⁱ]
yz_index  = [1]

output_slices = vcat(
    [(f, :xy, xy_levels) for f in xy_fields],
    [(f, :yz, yz_index)  for f in yz_fields],
)

output_dir = joinpath(@__DIR__, "output", jobid_procid)
mkpath(output_dir)

# ─── Phase 1: First loop — 64 steps at Δt=0.01 ──────────────────────
@info "[$rank] Phase 1: first loop (64 steps, Δt=0.01)" now(UTC)
@time "[$rank] first loop" compiled_loop!(model, Ninner, Δt_r)

local_nan_check(rank, "after first loop", model)

# ─── Phase 2: Second loop — 64 steps at Δt=0.01 ─────────────────────
@info "[$rank] Phase 2: second loop (64 steps, Δt=0.01)" now(UTC)
@time "[$rank] second loop" compiled_loop!(model, Ninner, Δt_r)

# ─── Phase 3: 8 warmup blocks × 256 steps (4×64) at Δt=0.01, with saves
const Nwarmup = 8
const Ncalls_per_block_warmup = 4   # 4 × 64 = 256 steps per block

@info "[$rank] Phase 3: $Nwarmup warmup blocks × $(Ncalls_per_block_warmup*64) steps (Δt=0.01)" now(UTC)
wall_start = time_ns()
for k in 1:Nwarmup
    t0 = time_ns()
    for _ in 1:Ncalls_per_block_warmup
        compiled_loop!(model, Ninner, Δt_r)
    end
    wall_block = (time_ns() - t0) / 1e9
    sim_time   = Ncalls_per_block_warmup * 64 * k * Δt
    total_wall = (time_ns() - wall_start) / 1e9

    @info @sprintf("[%d] warmup %d/%d  wall=%.1fs  sim=%.1fs  total_wall=%.0fs",
                    rank, k, Nwarmup, wall_block, sim_time, total_wall)

    #=
    block_dir = joinpath(output_dir, @sprintf("warmup_%04d", k))
    @time "[$rank] save warmup $k" begin
        GordonBell25.save_model_state(block_dir, model, arch;
            label = "output", slices = output_slices)
    end
    @info "[$rank] saved warmup $k" block_dir
    =#

    flush(stderr); flush(stdout)
end

block_dir = joinpath(output_dir, "warmup_1")
GordonBell25.save_model_state(block_dir, model, arch; label = "output", slices = output_slices)

local_nan_check(rank, "after warmup (8 blocks)", model)

# ─── Phase 4: Ramp Δt to 0.005, long production run ───────────────────
Δt_r = make_dt(0.005, local_arch, arch, Ndev)
@info "[$rank] Phase 4: bumped Δt to 0.05 (no recompile)" now(UTC)

const Nouter = 2000
const Ncalls_per_block = 4   # 4 × 64 = 256 steps per block
steps_per_block = Ncalls_per_block * 64

@info "[$rank] Starting production: $Nouter blocks × $steps_per_block steps (Δt=0.05)" now(UTC)

wall_start = time_ns()
for k in 1:Nouter
    t0 = time_ns()
    for _ in 1:Ncalls_per_block
        compiled_loop!(model, Ninner, Δt_r)
    end
    wall_block = (time_ns() - t0) / 1e9
    sim_time   = steps_per_block * k * 0.05
    total_wall = (time_ns() - wall_start) / 1e9
    sypd       = (steps_per_block * 0.05) / (365.25 * 86400 * wall_block) * 365.25

    @info @sprintf("[%d] block %d/%d  wall=%.1fs  sim=%.1fs  SYPD=%.5f  total_wall=%.0fs",
                    rank, k, Nouter, wall_block, sim_time, sypd, total_wall)

    #=
    block_dir = joinpath(output_dir, @sprintf("block_%04d", k))
    @time "[$rank] save block $k" begin
        GordonBell25.save_model_state(block_dir, model, arch;
            label = "output", slices = output_slices)
    end
    @info "[$rank] saved block $k" block_dir
    =#

    flush(stderr); flush(stdout)
end

block_dir = joinpath(output_dir, "block_1")
GordonBell25.save_model_state(block_dir, model, arch; label = "output", slices = output_slices)

# ─── Phase 5: Ramp Δt to 0.01, long production run ───────────────────
Δt_r = make_dt(0.01, local_arch, arch, Ndev)
@info "[$rank] Phase 4: bumped Δt to 0.05 (no recompile)" now(UTC)

const Nouter = 2000
const Ncalls_per_block = 4   # 4 × 64 = 256 steps per block
steps_per_block = Ncalls_per_block * 64

@info "[$rank] Starting production: $Nouter blocks × $steps_per_block steps (Δt=0.05)" now(UTC)

wall_start = time_ns()
for k in 1:Nouter
    t0 = time_ns()
    for _ in 1:Ncalls_per_block
        compiled_loop!(model, Ninner, Δt_r)
    end
    wall_block = (time_ns() - t0) / 1e9
    sim_time   = steps_per_block * k * 0.05
    total_wall = (time_ns() - wall_start) / 1e9
    sypd       = (steps_per_block * 0.05) / (365.25 * 86400 * wall_block) * 365.25

    @info @sprintf("[%d] block %d/%d  wall=%.1fs  sim=%.1fs  SYPD=%.5f  total_wall=%.0fs",
                    rank, k, Nouter, wall_block, sim_time, sypd, total_wall)

    #=
    block_dir = joinpath(output_dir, @sprintf("block_%04d", k))
    @time "[$rank] save block $k" begin
        GordonBell25.save_model_state(block_dir, model, arch;
            label = "output", slices = output_slices)
    end
    @info "[$rank] saved block $k" block_dir
    =#

    flush(stderr); flush(stdout)
end

block_dir = joinpath(output_dir, "block_2")
GordonBell25.save_model_state(block_dir, model, arch; label = "output", slices = output_slices)

# ─── Phase 5: Ramp Δt to 0.01, long production run ───────────────────
Δt_r = make_dt(0.05, local_arch, arch, Ndev)
@info "[$rank] Phase 4: bumped Δt to 0.05 (no recompile)" now(UTC)

const Nouter = 2000
const Ncalls_per_block = 4   # 4 × 64 = 256 steps per block
steps_per_block = Ncalls_per_block * 64

@info "[$rank] Starting production: $Nouter blocks × $steps_per_block steps (Δt=0.05)" now(UTC)

wall_start = time_ns()
for k in 1:Nouter
    t0 = time_ns()
    for _ in 1:Ncalls_per_block
        compiled_loop!(model, Ninner, Δt_r)
    end
    wall_block = (time_ns() - t0) / 1e9
    sim_time   = steps_per_block * k * 0.05
    total_wall = (time_ns() - wall_start) / 1e9
    sypd       = (steps_per_block * 0.05) / (365.25 * 86400 * wall_block) * 365.25

    @info @sprintf("[%d] block %d/%d  wall=%.1fs  sim=%.1fs  SYPD=%.5f  total_wall=%.0fs",
                    rank, k, Nouter, wall_block, sim_time, sypd, total_wall)

    block_dir = joinpath(output_dir, @sprintf("block_3_%04d", k))
    @time "[$rank] save block $k" begin
        GordonBell25.save_model_state(block_dir, model, arch;
            label = "output", slices = output_slices)
    end
    @info "[$rank] saved block $k" block_dir

    flush(stderr); flush(stdout)
end

