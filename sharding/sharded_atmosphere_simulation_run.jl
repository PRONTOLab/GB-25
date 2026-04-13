using Dates
@info "Starting sharded atmosphere simulation" now(UTC)

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors, is_distributed_env_present
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 64,
)

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
GordonBell25.initialize(; single_gpu_per_process=false)

local_arch = ReactantState()
arch = local_arch

Ndev = if arch isa ReactantState
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
    rank = if local_arch isa ReactantState
        Reactant.Distributed.local_rank()
    else
        comm = MPI.COMM_WORLD
        MPI.Comm_rank(comm)
    end
end

H = 4
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nλ = Tλ - 2H
Nφ = Tφ - 2H

Δt = 1.0
column_height = 30e3
sst_anomaly = parse(Float64, get(ENV, "SST_ANOMALY", "0"))

@info "Configuration" Nλ Nφ Nz Δt Rx Ry Ndev sst_anomaly

# ── IC file ────────────────────────────────────────────────────────────

ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                   "atmosphere_no_microphysics_1deg_14day.jld2")

if !isfile(ic_path)
    error("IC file not found at $ic_path")
end

@info "Using IC file" ic_path

# ── Build model ────────────────────────────────────────────────────────

@info "Building model..." now(UTC)
model = GordonBell25.moist_baroclinic_wave_model(arch;
    Nλ, Nφ, Nz,
    H = column_height,
    Δt,
    halo = (H, H, 4),
    cloud_formation_τ_relax = 30.0,
    initial_conditions_path = ic_path,
    interpolation_type = :linear,
    sst_anomaly,
)

@show model

# ── Field diagnostics (no CPU copy) ───────────────────────────────────

function report_state(model, label)
    fields = Oceananigans.fields(model)
    for name in keys(fields)
        f = fields[name]
        p = parent(f)
        mx = Float64(maximum(p))
        mn = Float64(minimum(p))
        @printf("  [%s] %6s: min=% .6e  max=% .6e\n", label, name, mn, mx)
        if isnan(mx) || isnan(mn)
            @error "NaN detected in $name at $label — aborting"
            exit(1)
        end
    end
end

@info "Initial state:"
report_state(model, "IC")

# ── Compile ────────────────────────────────────────────────────────────

Ninner = 1024

Ninner_r = if Ndev == 1
    ConcreteRNumber(Ninner)
else
    sharding = Sharding.NamedSharding(arch.connectivity, ())
    ConcreteRNumber(Ninner; sharding)
end

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all)

@info "Compiling first_time_step!..." now(UTC)
rfirst! = @time @compile compile_options=compile_options first_time_step!(model)

@info "Compiling loop! (Ninner=$Ninner)..." now(UTC)
compiled_loop! = @time @compile compile_options=compile_options loop!(model, Ninner_r)

# ── First time step ────────────────────────────────────────────────────

@info "Running first_time_step!..." now(UTC)
@time rfirst!(model)

report_state(model, "step 1")

# ── Main loop ──────────────────────────────────────────────────────────

Nouter = 10

@info "Starting outer loop: $Nouter blocks × $Ninner steps (Δt=$Δt)" now(UTC)

wall_start = time_ns()
for k in 1:Nouter
    t0 = time_ns()
    compiled_loop!(model, Ninner_r)
    wall_block = (time_ns() - t0) / 1e9
    total_steps = Ninner * k + 1
    sim_time = total_steps * Δt
    total_wall = (time_ns() - wall_start) / 1e9
    sypd = (Ninner * Δt) / (365.25 * 86400 * wall_block) * 365.25

    @info @sprintf("block %d/%d: %d steps, wall=%.1fs, sim_time=%.0fs, SYPD=%.4f, total_wall=%.0fs",
                    k, Nouter, total_steps, wall_block, sim_time, sypd, total_wall)

    report_state(model, @sprintf("step %d", total_steps))

    flush(stderr); flush(stdout)
end

@info "Done!" now(UTC)
