using ArgParse

const args_settings = ArgParseSettings()
@add_arg_table! args_settings begin
    "--grid-x"
        help = "Number of longitude grid points."
        default = 32
        arg_type = Int
    "--grid-y"
        help = "Number of latitude grid points."
        default = 16
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

using GordonBell25
using Oceananigans
default_float_type = if parsed_args["precision"] == 64
    Float64
elseif parsed_args["precision"] == 32
    Float32
else
    throw(AssertionError("Unknown precision $(parsed_args["precision"])"))
end
Oceananigans.defaults.FloatType = default_float_type
using CUDA
using Reactant

GordonBell25.preamble()

# ═══════════════════════════════════════════════════════════════════════════════
# Build models
# ═══════════════════════════════════════════════════════════════════════════════

throw_error = true
include_halos = true
rtol = sqrt(eps(default_float_type))
atol = 0

model_kw = (;
    Nλ   = parsed_args["grid-x"],
    Nφ   = parsed_args["grid-y"],
    Nz   = parsed_args["grid-z"],
    halo = (8, 8, 8),
    Δt   = 1e-9,
)

rarch = Oceananigans.Architectures.ReactantState()
varch = CPU()
rmodel = GordonBell25.moist_baroclinic_wave_model(rarch; model_kw...)
vmodel = GordonBell25.moist_baroclinic_wave_model(varch; model_kw...)
@show vmodel
@show rmodel

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Initial conditions (deterministic DCMIP-2016)
# Both constructors set ICs internally; comparing here verifies the
# Reactant-compiled IC kernels match the CPU path.
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

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all)
rfirst! = @compile compile_options=compile_options GordonBell25.first_time_step!(rmodel)
@showtime rfirst!(rmodel)
@showtime GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Multiple time steps
# ═══════════════════════════════════════════════════════════════════════════════

rstep! = @compile compile_options=compile_options GordonBell25.time_step!(rmodel)

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
@jit Oceananigans.TimeSteppers.update_state!(rmodel)
# Fix for above: wrap in compile and raise. TODO figure out why raise=false crashes because of TGammaOp

@info "After syncing and updating state again:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@showtime rloop!(rmodel, rNt)
@showtime GordonBell25.loop!(vmodel, Nt)

@info "After a loop of $(Nt) steps:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
