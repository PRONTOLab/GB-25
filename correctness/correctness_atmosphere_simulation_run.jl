using ArgParse
using GordonBell25
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 16,
)

default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type

using CUDA
using Reactant

GordonBell25.preamble()

# ═══════════════════════════════════════════════════════════════════════════════
# Build models
# ═══════════════════════════════════════════════════════════════════════════════

throw_error = false
include_halos = true
rtol = 2 * sqrt(eps(default_float_type))
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
rupdate! = @compile sync=true raise=true GordonBell25.update_state!(rmodel)
rupdate!(rmodel)
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
