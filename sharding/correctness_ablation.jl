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

local_arch = Oceananigans.ReactantState()
rarch = local_arch

Ndev = if rarch isa Oceananigans.ReactantState
    length(Reactant.devices())
else
    comm = MPI.COMM_WORLD
    MPI.Comm_size(comm)
end

@show Ndev

Rx, Ry = Ndev == 1 ? (1, 1) : GordonBell25.factors(Ndev)
@info "[correctness] partition" Ndev Rx Ry

if Ndev == 1
    rank = 0
else
    rarch = Oceananigans.Distributed(rarch; partition = Partition(Rx, Ry, 1))
    rank = if local_arch isa Oceananigans.ReactantState
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

# ═══════════════════════════════════════════════════════════════════════════════
# Ablation levels (cumulative):
#   1 = no microphysics
#   2 = no microphysics + no surface fluxes
#   3 = no microphysics + no surface fluxes + no advection
#   4 = no microphysics + no surface fluxes + no advection + no coriolis
#
# MICROPHYSICS_TYPE overrides microphysics config (independent of ablation):
#   warm_1m  = OneMomentCloudMicrophysics, warm phase only (no ice)
#   kessler  = DCMIP2016KesslerMicrophysics (warm rain)
#   full     = OneMomentCloudMicrophysics, mixed phase (default)
# ═══════════════════════════════════════════════════════════════════════════════

# Individual ablation via env vars (each independently toggles one component OFF):
#   ABLATE_MICROPHYSICS=1  → no microphysics
#   ABLATE_CORIOLIS=1      → no coriolis
#   ABLATE_ADVECTION=1     → no WENO advection
#   ABLATE_SURFACE=1       → no surface fluxes
# All default to ON if not set.

with_microphysics   = get(ENV, "ABLATE_MICROPHYSICS", "") != "1"
with_coriolis       = get(ENV, "ABLATE_CORIOLIS", "") != "1"
with_advection      = get(ENV, "ABLATE_ADVECTION", "") != "1"
with_surface_fluxes = get(ENV, "ABLATE_SURFACE", "") != "1"

@info "Model config" with_microphysics with_surface_fluxes with_advection with_coriolis

include_halos = false
throw_error   = false
rtol = 2 * sqrt(eps(default_float_type))
atol = 0
column_height = 30e3

model_kw = (;
    Nλ, Nφ, Nz,
    halo = (4, 4, 4),
    Δt   = 0.5,
    H    = column_height,
    initial_conditions_path = nothing,
    with_microphysics,
    with_surface_fluxes,
    with_advection,
    with_coriolis,
    sst_anomaly = parse(Float64, get(ENV, "SST_ANOMALY", "0")),
)

varch = CPU()

@info "Building vanilla CPU model..."
vmodel = GordonBell25.moist_baroclinic_wave_model(varch; model_kw...)
@info "Building Reactant model..."
rmodel = GordonBell25.moist_baroclinic_wave_model(rarch; model_kw...)
@show vmodel
@show rmodel

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Compare analytic ICs
# ═══════════════════════════════════════════════════════════════════════════════

@info "After construction (analytic DCMIP-2016 ICs):"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — First time step
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
