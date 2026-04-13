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
atol = eps(default_float_type)

model_kw = (;
    Nλ   = parsed_args["grid-x"],
    Nφ   = parsed_args["grid-y"],
    Nz   = parsed_args["grid-z"],
    halo = (8, 8, 8),
    Δt   = 1e-9,
    with_microphysics = false,
    initial_conditions_path = nothing,
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
# Stage 2 — First time step (verbose, using ported RK3)
# Sync CPU → Reactant to start from an exact match, then step both.
# ═══════════════════════════════════════════════════════════════════════════════

GordonBell25.atmos_sync_states!(rmodel, vmodel)
GordonBell25.atmos_zero_tendencies!(rmodel)
GordonBell25.atmos_zero_tendencies!(vmodel)

Δt = vmodel.clock.last_Δt

@info "Running update_state! on both models before first step..."
Oceananigans.TimeSteppers.update_state!(vmodel)
Oceananigans.TimeSteppers.update_state!(rmodel)

@info "After update_state!:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Running time_step_verbose! on both models..."
GordonBell25.time_step_verbose!(rmodel, Δt)
GordonBell25.time_step_verbose!(vmodel, Δt)

@info "After first verbose time step:"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Multiple time steps with per-step comparison
# ═══════════════════════════════════════════════════════════════════════════════

Nt = 10
@info "Running $Nt verbose time steps with per-step comparison..."
for n in 1:Nt
    GordonBell25.time_step_verbose!(rmodel, Δt)
    GordonBell25.time_step_verbose!(vmodel, Δt)
    @info "After step $n / $Nt:"
    GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
end
