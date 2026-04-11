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
#
# GordonBell25.initialize(; single_gpu_per_process=false)

local_arch = Oceananigans.ReactantState()
rarch = local_arch

Ndev = if rarch isa Oceananigans.ReactantState
    length(Reactant.devices())
 else
    comm = MPI.COMM_WORLD
    MPI.Comm_size(comm)
 end
 
 @show Ndev
 
 Rx, Ry = 2, 2
 
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
 
 @info "[$rank] allocations" GordonBell25.allocatorstats()
 
 H = 4
 Tλ = parsed_args["grid-x"] * Rx
 Tφ = parsed_args["grid-y"] * Ry
 Nz = parsed_args["grid-z"]
 
 Nλ = Tλ - 2H
 Nφ = Tφ - 2H

# ═══════════════════════════════════════════════════════════════════════════════
# IC file — required for this test
# ═══════════════════════════════════════════════════════════════════════════════

ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                   "atmosphere_coarsened_32x16x8.jld2")
if !isfile(ic_path)
    error("IC file not found at $ic_path — this test requires file-based ICs")
end
@info "Using IC file" ic_path

# ═══════════════════════════════════════════════════════════════════════════════
# Build models — Reactant (InterpolateArray Nearest) vs vanilla (KA kernel NN)
# ═══════════════════════════════════════════════════════════════════════════════

include_halos = false
throw_error   = false
rtol = 2 * sqrt(eps(default_float_type))
atol = 0
column_height = 30e3

model_kw = (;
    Nλ   = Nλ,
    Nφ   = Nφ,
    Nz   = Nz,
    halo = (4, 4, 4),
    Δt   = 0.5,
    initial_conditions_path = ic_path,
    H=column_height,
    cloud_formation_τ_relax=10.0
)

varch = CPU()

@info "Building vanilla CPU model..."
vmodel = GordonBell25.moist_baroclinic_wave_model(varch; model_kw...)
@info "Building Reactant model..."
rmodel = GordonBell25.moist_baroclinic_wave_model(rarch; model_kw...)
@show vmodel
@show rmodel

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Compare ICs after file-based interpolation
# Reactant used InterpolateArray(Nearest), vanilla used KA _nn_atmos_field_copy!
# ═══════════════════════════════════════════════════════════════════════════════

@info "After file-based IC loading (Nearest-neighbor interpolation):"
GordonBell25.atmos_compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — First time step
# Sync CPU ← Reactant so both start from an exact match, then step both.
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
