using ArgParse
using GordonBell25
using Oceananigans
using CUDA
using Reactant

GordonBell25.preamble()

const args_settings = ArgParseSettings()

@add_arg_table! args_settings begin
    "--grid-x"
        help = "Per-device grid points in longitude (total = grid-x * Rx)."
        default = 32
        arg_type = Int
    "--grid-y"
        help = "Per-device grid points in latitude (total = grid-y * Ry)."
        default = 32
        arg_type = Int
    "--grid-z"
        help = "Number of vertical grid points."
        default = 4
        arg_type = Int
    "--precision"
        help = "Number of bits of precision"
        default = 64
        arg_type = Int
end

const parsed_args = parse_args(ARGS, args_settings)

default_float_type = if parsed_args["precision"] == 64
    Float64
elseif parsed_args["precision"] == 32
    Float32
else
    throw(AssertionError("Unknown precision $(parsed_args["precision"])"))
end

Oceananigans.defaults.FloatType = default_float_type

if !GordonBell25.is_distributed_env_present()
    using MPI
    MPI.Init()
end

throw_error = true
include_halos = true
rtol = sqrt(eps(default_float_type))
atol = 0

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

Rx, Ry = GordonBell25.factors(Ndev)

rarch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition = Partition(Rx, Ry, 1)
)

rank = Reactant.Distributed.local_rank()

H = 8
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nλ = Tλ - 2H
Nφ = Tφ - 2H

@info "[$rank] Grid: Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz (Rx=$Rx, Ry=$Ry, Ndev=$Ndev)"

model_kw = (;
    Nλ, Nφ, Nz,
    halo = (H, H, H),
    Δt   = 1e-9,
)

varch = CPU()
rmodel = GordonBell25.moist_baroclinic_wave_model(rarch; model_kw...)
vmodel = GordonBell25.moist_baroclinic_wave_model(varch; model_kw...)
@show vmodel
@show rmodel
@assert rmodel.architecture isa Distributed

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Initial conditions (deterministic DCMIP-2016)
# Both constructors set ICs internally; comparing here verifies the
# Reactant-compiled IC kernels match the CPU path under sharding.
# ═══════════════════════════════════════════════════════════════════════════════

@info "After construction (deterministic DCMIP-2016 ICs):"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — First time step
# Sync CPU → Reactant to start from an exact match, then step both.
# ═══════════════════════════════════════════════════════════════════════════════

GordonBell25.atmos_sync_states!(rmodel, vmodel)
GordonBell25.atmos_zero_tendencies!(rmodel)
GordonBell25.atmos_zero_tendencies!(vmodel)
rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
@showtime rfirst!(rmodel)
@showtime GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Multiple time steps
# ═══════════════════════════════════════════════════════════════════════════════

rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)

@info "Warm up:"
@showtime rstep!(rmodel)
@showtime rstep!(rmodel)
@showtime GordonBell25.time_step!(vmodel)
@showtime GordonBell25.time_step!(vmodel)

Nt = 10
@info "Time step with Reactant:"
for _ in 1:Nt
    @showtime rstep!(rmodel)
end

@info "Time step vanilla:"
for _ in 1:Nt
    @showtime GordonBell25.time_step!(vmodel)
end

@info "After $(Nt) steps:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Loop (traced for-loop)
# Re-sync so any accumulated drift doesn't mask loop-specific issues.
# ═══════════════════════════════════════════════════════════════════════════════

GordonBell25.atmos_sync_states!(rmodel, vmodel)
rupdate! = @compile sync=true raise=true GordonBell25.update_state!(rmodel)
rupdate!(rmodel)

@info "After syncing and updating state again:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@showtime rloop!(rmodel, rNt)
@showtime GordonBell25.loop!(vmodel, Nt)

@info "After a loop of $(Nt) steps:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
