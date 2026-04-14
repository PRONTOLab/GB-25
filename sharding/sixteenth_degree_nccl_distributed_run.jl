# Vanilla-Oceananigans 1/16° atmosphere run — 8 GPUs with NCCLDistributed.
# Initialized from 1/8° checkpoint_step_008193 (interpolated 2× in horizontal)
# with cloud-field IC. No IC-relaxation needed (2× upsample is mild).
# Latitude range (-80, 80), Δt=0.8s, SST anomaly = +2 K, τ_cloud=120s.
#
# Launch: ~/.julia/bin/mpiexecjl -n 8 --project julia -O0 sharding/sixteenth_degree_nccl_distributed_run.jl
#
# IMPORTANT: Does NOT load GordonBell25 or Reactant on this box.
# `RelaxToICForcing` is inlined below instead of imported from GordonBell25,
# because `using GordonBell25` pulls Reactant which corrupts CUDA.jl PTX here.

using Dates
using MPI
MPI.Init()

rank   = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

using JLD2
using Printf
using CUDA
using NCCL

CUDA.device!(rank % length(CUDA.devices()))

rank == 0 && @info "Starting NCCL-distributed 1/16° atmosphere simulation" nprocs now(UTC)
rank == 0 && @info "GPU assignment" rank gpu=CUDA.device()

using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.DistributedComputations: Distributed, Partition
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using KernelAbstractions: @kernel, @index
import CUDA: Adapt
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics, ExplicitTimeStepping
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density, specific_prognostic_moisture
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics

Oceananigans.defaults.FloatType = Float32

const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributed = NCCLExt.NCCLDistributed

# ═══════════════════════════════════════════════════════════════════════════
# DCMIP-2016 balanced reference state (used only for
#   (a) dynamics.reference_potential_temperature,
#   (b) surface_temperature in bulk flux BCs).
# The ACTUAL IC comes from the file; we do NOT call set_moist_baroclinic_wave!.
# ═══════════════════════════════════════════════════════════════════════════

const earth_radius   = 6371220.0
const gravity        = 9.80616
const Rd_dry         = 287.0
const ε_virtual      = 0.608
const κ_exponent     = 2.0 / 7.0
const p_ref          = 1e5

const T_equator      = 310.0
const T_polar        = 240.0
const T_mean         = 0.5 * (T_equator + T_polar)
const lapse_rate     = 0.005
const jet_width      = 3.0
const vert_width     = 2.0

const coeff_A        = 1.0 / lapse_rate
const coeff_B        = (T_mean - T_polar) / (T_mean * T_polar)
const coeff_C        = 0.5 * (jet_width + 2) * (T_equator - T_polar) / (T_equator * T_polar)
const scale_height   = Rd_dry * T_mean / gravity

const q0_surface    = 0.018
const φ_width       = 2π / 9
const p_width       = 34000.0
const η_tropopause  = 0.1
const q_tropopause  = 1e-12

function vertical_structure(z)
    ζ      = z / (vert_width * scale_height)
    exp_ζ² = exp(-ζ^2)
    τ₁ = coeff_A * lapse_rate / T_mean * exp(lapse_rate * z / T_mean) +
         coeff_B * (1 - 2 * ζ^2) * exp_ζ²
    τ₂ = coeff_C * (1 - 2 * ζ^2) * exp_ζ²
    I₁ = coeff_A * (exp(lapse_rate * z / T_mean) - 1) +
         coeff_B * z * exp_ζ²
    I₂ = coeff_C * z * exp_ζ²
    return (; τ₁, τ₂, I₁, I₂)
end

F_temperature(cosφ) = cosφ^jet_width - jet_width / (jet_width + 2) * cosφ^(jet_width + 2)

function virtual_temperature(φ, z)
    vs = vertical_structure(z)
    return 1.0 / (vs.τ₁ - vs.τ₂ * F_temperature(cos(φ)))
end

function balanced_pressure(φ, z)
    vs = vertical_structure(z)
    return p_ref * exp(-gravity / Rd_dry * (vs.I₁ - vs.I₂ * F_temperature(cos(φ))))
end

function moisture_profile(φ, z)
    p = balanced_pressure(φ, z)
    η = p / p_ref
    q_below = q0_surface * exp(-(φ / φ_width)^4) *
                           exp(-((η - 1) * p_ref / p_width)^2)
    return ifelse(η > η_tropopause, q_below, q_tropopause)
end

function initial_theta(λ_deg, φ_deg, z)
    φ  = deg2rad(φ_deg)
    Tv = virtual_temperature(φ, z)
    p  = balanced_pressure(φ, z)
    q  = moisture_profile(φ, z)
    T  = Tv / (1 + ε_virtual * q)
    return T * (p_ref / p)^κ_exponent
end

theta_reference(z)               = initial_theta(0.0, 0.0, z)
surface_temperature(λ_deg, φ_deg) = virtual_temperature(deg2rad(φ_deg), 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# Relaxation-to-IC forcing (inlined from src/relaxation_to_ic.jl).
# 4-field variant: damps {ρu, ρv, ρθ, ρqᵛ} only, no ρ / ρw / cloud snapshots.
# ═══════════════════════════════════════════════════════════════════════════

struct RelaxToICForcing{Name, F, T}
    ic       :: F
    α0       :: T
    T_decay  :: T
end

RelaxToICForcing(name::Symbol, ic, α0::Real, T_decay::Real) =
    let Tpromoted = promote_type(typeof(α0), typeof(T_decay))
        RelaxToICForcing{name, typeof(ic), Tpromoted}(ic, Tpromoted(α0), Tpromoted(T_decay))
    end

@inline function (f::RelaxToICForcing{Name})(i, j, k, grid, clock, fields) where {Name}
    t = clock.time
    α = max(zero(t), f.α0 * (1 - t / f.T_decay))
    @inbounds ic_val  = f.ic[i, j, k]
    @inbounds cur_val = fields[Name][i, j, k]
    return -α * (cur_val - ic_val)
end

# When the kernel is adapted for the GPU, strip the Field stored in `ic`
# down to its underlying OffsetArray-of-CuDeviceArray so the whole forcing
# struct is isbits-compatible. Without this, the Field's grid carries the
# (non-isbits) NCCL communicator and CUDA stream handles into the kernel.
function Adapt.adapt_structure(to, f::RelaxToICForcing{Name}) where {Name}
    ic_a = Adapt.adapt(to, f.ic)
    return RelaxToICForcing{Name, typeof(ic_a), typeof(f.α0)}(ic_a, f.α0, f.T_decay)
end

function build_ic_relaxation_forcing_5field(grid; α0, T_decay)
    snapshots = (
        ρu   = XFaceField(grid),
        ρv   = YFaceField(grid),
        ρw   = ZFaceField(grid),
        ρθ   = CenterField(grid),
        ρqᵛ  = CenterField(grid),
    )
    forcing = (
        ρu  = RelaxToICForcing(:ρu,  snapshots.ρu,  α0, T_decay),
        ρv  = RelaxToICForcing(:ρv,  snapshots.ρv,  α0, T_decay),
        ρw  = RelaxToICForcing(:ρw,  snapshots.ρw,  α0, T_decay),
        ρθ  = RelaxToICForcing(:ρθ,  snapshots.ρθ,  α0, T_decay),
        ρqᵛ = RelaxToICForcing(:ρqᵛ, snapshots.ρqᵛ, α0, T_decay),
    )
    return forcing, snapshots
end

# ═══════════════════════════════════════════════════════════════════════════
# Damp-to-zero forcing: same discrete tendency style as RelaxToICForcing but
# with a constant target of 0. Use on cloud condensate (ρqᶜˡ, ρqᶜⁱ) to
# prevent the fast microphysics ↔ dynamics feedback loop from spinning up
# during the IC staircase adjustment.
#
#     F(i,j,k) = -α(t) * fields[Name][i,j,k]
#     α(t) = max(0, α0 * (1 - t/T_decay))
#
# No Field/ic snapshot → no Adapt needed; the struct is already isbits.
# ═══════════════════════════════════════════════════════════════════════════

struct DampToZeroForcing{Name, T}
    α0      :: T
    T_decay :: T
end

DampToZeroForcing(name::Symbol, α0::Real, T_decay::Real) =
    let Tp = promote_type(typeof(α0), typeof(T_decay))
        DampToZeroForcing{name, Tp}(Tp(α0), Tp(T_decay))
    end

@inline function (f::DampToZeroForcing{Name})(i, j, k, grid, clock, fields) where {Name}
    t = clock.time
    α = max(zero(t), f.α0 * (1 - t / f.T_decay))
    @inbounds cur_val = fields[Name][i, j, k]
    return -α * cur_val
end

function build_cloud_damping_forcing(; α0, T_decay)
    return (
        ρqᶜˡ = DampToZeroForcing(:ρqᶜˡ, α0, T_decay),
        ρqᶜⁱ = DampToZeroForcing(:ρqᶜⁱ, α0, T_decay),
    )
end

function copy_ic_snapshots_5field!(snapshots, model)
    Oceananigans.set!(snapshots.ρu,  model.momentum.ρu)
    Oceananigans.set!(snapshots.ρv,  model.momentum.ρv)
    Oceananigans.set!(snapshots.ρw,  model.momentum.ρw)
    Oceananigans.set!(snapshots.ρθ,  model.formulation.potential_temperature_density)
    Oceananigans.set!(snapshots.ρqᵛ, model.moisture_density)
    # CRITICAL: Snapshot fields are freshly allocated, so their halo regions
    # contain uninitialized GPU memory (often NaN). The relaxation forcing
    # reads f.ic[i,j,k] inside tendency kernels at indices that can land in
    # halos via offset arithmetic. Without this call, NaN halos contaminate
    # the momentum/thermo tendencies. Symptom previously observed: r1, r3
    # NaN at iter 2 in ρu/ρv/ρw/ρθ while ρqv stays intact (memory-layout luck).
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρu)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρv)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρw)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρθ)
    Oceananigans.BoundaryConditions.fill_halo_regions!(snapshots.ρqᵛ)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# IC loader — each rank opens the full 1/8° checkpoint, builds a GPU source
# grid at 1/8° resolution over (-80, 80), and interpolate!s into its local
# 1/32° portion.
# ═══════════════════════════════════════════════════════════════════════════

function load_ic_distributed!(model, path::String;
                              H = 30e3,
                              src_latitude  = (-80, 80),
                              src_longitude = (0, 360))
    Nλ_src, Nφ_src, Nz_src, ρ_data, ρu_data, ρv_data, ρw_data, ρθ_data, ρqv_data =
        JLD2.jldopen(path, "r") do file
            (file["Nλ"], file["Nφ"], file["Nz"],
             file["ρ"], file["ρu"], file["ρv"], file["ρw"],
             file["ρθ"], file["ρqᵛ"])
        end

    # Optionally pick up cloud / rain / snow condensate fields if the
    # checkpoint was written with them (new format).
    ρqcl_data, ρqci_data, ρqʳ_data, ρqˢ_data = JLD2.jldopen(path, "r") do file
        ρqcl = haskey(file, "micro_ρqᶜˡ") ? file["micro_ρqᶜˡ"] : nothing
        ρqci = haskey(file, "micro_ρqᶜⁱ") ? file["micro_ρqᶜⁱ"] : nothing
        ρqʳ  = haskey(file, "ρqʳ")        ? file["ρqʳ"]         : nothing
        ρqˢ  = haskey(file, "ρqˢ")        ? file["ρqˢ"]         : nothing
        (ρqcl, ρqci, ρqʳ, ρqˢ)
    end

    grid = model.grid
    arch = Oceananigans.architecture(grid)
    FT   = eltype(grid)

    pairs = Any[
        (ρ_data,   dynamics_density(model.dynamics)),
        (ρu_data,  model.momentum.ρu),
        (ρv_data,  model.momentum.ρv),
        (ρw_data,  model.momentum.ρw),
        (ρθ_data,  model.formulation.potential_temperature_density),
        (ρqv_data, model.moisture_density),
    ]

    if ρqcl_data !== nothing
        push!(pairs, (ρqcl_data, model.microphysical_fields[:ρqᶜˡ]))
        rank == 0 && @info "Loading micro_ρqᶜˡ" extrema=extrema(ρqcl_data)
    end
    if ρqci_data !== nothing
        push!(pairs, (ρqci_data, model.microphysical_fields[:ρqᶜⁱ]))
        rank == 0 && @info "Loading micro_ρqᶜⁱ" extrema=extrema(ρqci_data)
    end
    if ρqʳ_data !== nothing
        push!(pairs, (ρqʳ_data, model.microphysical_fields[:ρqʳ]))
        rank == 0 && @info "Loading ρqʳ" extrema=extrema(ρqʳ_data)
    end
    if ρqˢ_data !== nothing
        push!(pairs, (ρqˢ_data, model.microphysical_fields[:ρqˢ]))
        rank == 0 && @info "Loading ρqˢ" extrema=extrema(ρqˢ_data)
    end

    halo = Oceananigans.halo_size(grid)

    src_grid = LatitudeLongitudeGrid(GPU();
        size = (Nλ_src, Nφ_src, Nz_src),
        halo = halo,
        latitude  = src_latitude,
        longitude = src_longitude,
        z = (0, H))

    for (src_array, target_field) in pairs
        loc = Oceananigans.location(target_field)
        iloc = map(L -> L(), loc)
        src_field = Field(iloc, src_grid)
        gpu_data = Oceananigans.on_architecture(GPU(), Array{FT}(src_array))
        copyto!(Oceananigans.interior(src_field), gpu_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(src_field)

        rank == 0 && @info "interpolate!" field=nameof(typeof(target_field)) loc src=size(Oceananigans.interior(src_field)) dst=size(Oceananigans.interior(target_field))
        Oceananigans.Fields.interpolate!(target_field, src_field)
    end

    # Clamp all moisture/condensate fields to be non-negative. Source IC can
    # contain small negative values (~ -1e-4) from bounds-preserving WENO in
    # the source run; bilinear interpolation preserves them, and WENO +
    # microphysics amplify to NaN in the first step.
    function clamp_nonneg!(f, name)
        int = Oceananigans.interior(f)
        int .= max.(int, zero(FT))
        Oceananigans.BoundaryConditions.fill_halo_regions!(f)
        rank == 0 && @info "clamped $name to ≥ 0"
    end
    clamp_nonneg!(model.moisture_density, "ρqᵛ")
    if haskey(model.microphysical_fields, :ρqᶜˡ)
        clamp_nonneg!(model.microphysical_fields[:ρqᶜˡ], "ρqᶜˡ")
    end
    if haskey(model.microphysical_fields, :ρqᶜⁱ)
        clamp_nonneg!(model.microphysical_fields[:ρqᶜⁱ], "ρqᶜⁱ")
    end
    if haskey(model.microphysical_fields, :ρqʳ)
        clamp_nonneg!(model.microphysical_fields[:ρqʳ], "ρqʳ")
    end
    if haskey(model.microphysical_fields, :ρqˢ)
        clamp_nonneg!(model.microphysical_fields[:ρqˢ], "ρqˢ")
    end

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Model construction
# ═══════════════════════════════════════════════════════════════════════════

function build_atmosphere_model(arch; Nλ, Nφ, Nz, H, Δt, halo, latitude,
                                sst_anomaly::Real = 0,
                                relaxation::Union{Nothing,NTuple{2,Real}} = nothing,
                                cloud_damping::Union{Nothing,NTuple{2,Real}} = nothing)
    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nλ, Nφ, Nz),
                                 halo,
                                 longitude = (0, 360),
                                 latitude,
                                 z = (0, H))

    coriolis = SphericalCoriolis()

    dynamics = CompressibleDynamics(
        ExplicitTimeStepping();
        surface_pressure = p_ref,
        reference_potential_temperature = theta_reference,
    )

    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    FT = Oceananigans.defaults.FloatType
    # Slower condensation/freezing than the 30s used at 1/4° parent.
    rate = FT(1) / FT(120)
    cf = Breeze.Microphysics.ConstantRateCondensateFormation{FT}(rate)
    cloud_formation = NonEquilibriumCloudFormation(cf, cf)
    microphysics = ext.OneMomentCloudMicrophysics(; cloud_formation)

    weno     = WENO(order = 5)
    weno_pos = WENO(order = 5, bounds = (0, 1))
    momentum_advection = weno
    scalar_advection = (ρθ   = weno,
                        ρqᵛ  = weno_pos,
                        ρqᶜˡ = weno_pos,
                        ρqᶜⁱ = weno_pos,
                        ρqʳ  = weno_pos,
                        ρqˢ  = weno_pos)

    Cᴰ = 1e-3
    Uᵍ = 1e-2
    T₀ = (λ, φ) -> surface_temperature(λ, φ) + sst_anomaly

    ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

    boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)

    # Relaxation forcing: allocate snapshot fields on the distributed grid
    # BEFORE building the model so we can wire them into `forcing = ...`.
    ic_relax_forcing, ic_snapshots = if relaxation === nothing
        NamedTuple(), nothing
    else
        α0, T_decay = relaxation
        build_ic_relaxation_forcing_5field(grid; α0, T_decay)
    end

    # Optional cloud-condensate damping (target = 0) to suppress the fast
    # microphysics ↔ dynamics feedback loop during spinup.
    cloud_damp_forcing = if cloud_damping === nothing
        NamedTuple()
    else
        α0, T_decay = cloud_damping
        build_cloud_damping_forcing(; α0, T_decay)
    end

    merged_forcing = merge(ic_relax_forcing, cloud_damp_forcing)

    model = AtmosphereModel(grid; dynamics, coriolis, momentum_advection,
                            scalar_advection, microphysics, boundary_conditions,
                            forcing = merged_forcing)

    FT_grid = eltype(grid)
    model.clock.last_Δt = FT_grid(Δt)

    return model, ic_snapshots
end

function any_nan(model)
    fields = (dynamics_density(model.dynamics),
              model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
              model.formulation.potential_temperature_density,
              model.moisture_density)
    for f in fields
        any(isnan, parent(f)) && return true
    end
    return false
end

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

Rx = 4
Ry = 2
@assert nprocs == Rx * Ry "Expected $(Rx * Ry) MPI ranks, got $nprocs"

rank == 0 && @info "Setting up NCCLDistributed" Rx Ry
arch = NCCLDistributed(GPU(); partition = Partition(Rx, Ry, 1))

Nλ = 5760
Nφ = 2560
Nz = 64
column_height = 30e3
Δt = 1.2
sst_anomaly = 2.0

# IC-relaxation and cloud-condensate damping both OFF for this run.
relaxation = nothing
cloud_damping = nothing

ic_path = joinpath(@__DIR__, "..", "simulations", "initial_conditions",
                   "checkpoint_step_008193.jld2")
isfile(ic_path) || error("IC file not found at $ic_path")
rank == 0 && @info "IC file (1/8° checkpoint_step_008193 with cloud fields → 1/16°)" ic_path

rank == 0 && @info "Building model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, lat=(-80,80), Δt=$(Δt)s, sst_anomaly=$(sst_anomaly)K, relaxation=$relaxation, cloud_damping=$cloud_damping)..." now(UTC)
@time "build model" model, ic_snapshots = build_atmosphere_model(arch;
                                                                  Nλ, Nφ, Nz,
                                                                  H=column_height, Δt,
                                                                  halo=(4, 4, 4),
                                                                  latitude=(-80, 80),
                                                                  sst_anomaly,
                                                                  relaxation,
                                                                  cloud_damping)
rank == 0 && @show model

rank == 0 && @info "Loading ICs (interpolating 1/8° → 1/16°)..." now(UTC)
@time "load ICs" load_ic_distributed!(model, ic_path;
                                       H=column_height,
                                       src_latitude=(-80, 80),
                                       src_longitude=(0, 360))

if any_nan(model)
    error("NaN after IC load on rank $rank")
end
rank == 0 && @info "Post-load: no NaN"

# Per-rank extrema of the JUST-INTERPOLATED state (before any time step).
# This tells us whether failing ranks already had bad IC values, or whether
# the failure is dynamics-induced.
let m = model
    ip(f) = Oceananigans.interior(f)
    pp(f) = parent(f)
    function show_field(name, f)
        @info @sprintf("[r%d] post-IC %s  interior=[%+.4e,%+.4e]  parent=[%+.4e,%+.4e]",
                       rank, name,
                       Float64(minimum(ip(f))), Float64(maximum(ip(f))),
                       Float64(minimum(pp(f))), Float64(maximum(pp(f))))
    end
    show_field("ρ",   dynamics_density(m.dynamics))
    show_field("ρu",  m.momentum.ρu)
    show_field("ρv",  m.momentum.ρv)
    show_field("ρw",  m.momentum.ρw)
    show_field("ρθ",  m.formulation.potential_temperature_density)
    show_field("ρqᵛ", m.moisture_density)
end
MPI.Barrier(MPI.COMM_WORLD)

# Freeze snapshot fields to the just-interpolated state.
if ic_snapshots !== nothing
    @time "copy IC snapshots" copy_ic_snapshots_5field!(ic_snapshots, model)
    α0_r, Td_r = relaxation
    rank == 0 && @info "IC-relaxation forcing active (5-field: ρu, ρv, ρw, ρθ, ρqᵛ)" α0=α0_r T_decay=Td_r

    # Verify snapshot interior + parent (halo-included) extrema. Any NaN/Inf
    # in `parent(...)` after copy means a halo-fill bug we still need to fix.
    for k in keys(ic_snapshots)
        s = ic_snapshots[k]
        ip = Oceananigans.interior(s)
        pp = parent(s)
        @info @sprintf("[r%d] snapshot %s  interior=[%+.3e, %+.3e]  parent(incl.halos)=[%+.3e, %+.3e]",
                       rank, string(k),
                       Float64(minimum(ip)), Float64(maximum(ip)),
                       Float64(minimum(pp)), Float64(maximum(pp)))
    end
end

# ── First time step ────────────────────────────────────────────────────

rank == 0 && @info "First time step..." now(UTC)
@time "first step" begin
    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.TimeSteppers.time_step!(model, Δt)
end

if any_nan(model)
    error("NaN after first time step on rank $rank")
end
rank == 0 && @info "First step complete" model.clock

# Per-rank extrema RIGHT AFTER the manual first step (iter 1). Lets us see
# whether the ρw amplification kick happens at iter 1 or iter 2.
let m = model
    function show_field(name, f)
        ip = Oceananigans.interior(f)
        @info @sprintf("[r%d] iter1 %s  interior=[%+.4e,%+.4e]",
                       rank, name,
                       Float64(minimum(ip)), Float64(maximum(ip)))
    end
    show_field("ρ",   dynamics_density(m.dynamics))
    show_field("ρu",  m.momentum.ρu)
    show_field("ρv",  m.momentum.ρv)
    show_field("ρw",  m.momentum.ρw)
    show_field("ρθ",  m.formulation.potential_temperature_density)
    show_field("ρqᵛ", m.moisture_density)
end
MPI.Barrier(MPI.COMM_WORLD)

# ── Simulation setup ────────────────────────────────────────────────────

output_dir = joinpath(@__DIR__, "..", "simulations", "output", "nccl_8gpu_16th_deg")
rank == 0 && mkpath(output_dir)
MPI.Barrier(MPI.COMM_WORLD)

output_prefix = joinpath(output_dir, "fields_rank$rank")

stop_iter = 36000  # 12h sim at Δt=1.2

simulation = Simulation(model; Δt, stop_iteration=stop_iter)

# Custom per-rank JLD2 output (bypasses Oceananigans JLD2Writer which
# doesn't add rank suffixes for NCCLDistributed + AtmosphereModel).
output_interval = 9000  # write every 3h sim (4 writes over 12h)
function save_fields(sim)
    iter = sim.model.clock.iteration
    if iter % output_interval == 0
        m = sim.model
        filepath = output_prefix * "_iter$(lpad(iter, 6, '0')).jld2"
        JLD2.jldopen(filepath, "w") do file
            file["iteration"] = iter
            file["time"] = m.clock.time
            file["Δt"] = Δt
            file["ρ"]  = Array(Oceananigans.interior(dynamics_density(m.dynamics)))
            file["ρu"] = Array(Oceananigans.interior(m.momentum.ρu))
            file["ρv"] = Array(Oceananigans.interior(m.momentum.ρv))
            file["ρw"] = Array(Oceananigans.interior(m.momentum.ρw))
            file["ρθ"] = Array(Oceananigans.interior(m.formulation.potential_temperature_density))
            file["ρqᵛ"] = Array(Oceananigans.interior(m.moisture_density))
            for name in keys(m.microphysical_fields)
                file[string(name)] = Array(Oceananigans.interior(m.microphysical_fields[name]))
            end
        end
        @info "Saved rank $rank output to $filepath"
    end
end

simulation.callbacks[:save_fields] = Callback(save_fields, IterationInterval(output_interval))

# Per-iteration diagnostic: report extrema for the prognostic fields and bail
# the moment any NaN appears anywhere. Cheap-ish (one reduce per field per rank).
function field_extrema(f)
    p = Oceananigans.interior(f)
    return Float64(minimum(p)), Float64(maximum(p))
end

wall_start = Ref(time_ns())
function diagnostics(sim)
    m = sim.model
    ρ_min, ρ_max     = field_extrema(dynamics_density(m.dynamics))
    ρu_min, ρu_max   = field_extrema(m.momentum.ρu)
    ρv_min, ρv_max   = field_extrema(m.momentum.ρv)
    ρw_min, ρw_max   = field_extrema(m.momentum.ρw)
    ρθ_min, ρθ_max   = field_extrema(m.formulation.potential_temperature_density)
    ρqv_min, ρqv_max = field_extrema(m.moisture_density)

    nan_here = !isfinite(ρ_min)  || !isfinite(ρ_max)  ||
               !isfinite(ρu_min) || !isfinite(ρu_max) ||
               !isfinite(ρv_min) || !isfinite(ρv_max) ||
               !isfinite(ρw_min) || !isfinite(ρw_max) ||
               !isfinite(ρθ_min) || !isfinite(ρθ_max) ||
               !isfinite(ρqv_min) || !isfinite(ρqv_max)

    # Print this rank's extrema EVERY iter (so we can diagnose which rank/tile
    # blows up first; rank 0 alone tells us nothing about other tiles).
    wall = (time_ns() - wall_start[]) / 1e9
    @info @sprintf("[r%d] iter %5d  t=%6.2fs  wall=%6.1fs  ρ=[%+.3e,%+.3e] ρu=[%+.2e,%+.2e] ρv=[%+.2e,%+.2e] ρw=[%+.2e,%+.2e] ρθ=[%+.3e,%+.3e] ρqv=[%+.3e,%+.3e]",
                   rank, m.clock.iteration, m.clock.time, wall,
                   ρ_min, ρ_max, ρu_min, ρu_max, ρv_min, ρv_max,
                   ρw_min, ρw_max, ρθ_min, ρθ_max, ρqv_min, ρqv_max)
    if nan_here
        @error "NaN/Inf detected at iter $(m.clock.iteration) on rank $rank — see preceding [r$rank] line for the bad extrema"
        error("NaN/Inf detected at iter $(m.clock.iteration) on rank $rank")
    end
    flush(stderr); flush(stdout)
end

simulation.callbacks[:diagnostics] = Callback(diagnostics, IterationInterval(100))

rank == 0 && @info "Starting simulation (stop_iter=$stop_iter, Δt=$Δt)" now(UTC)
wall_start[] = time_ns()
Oceananigans.run!(simulation)

rank == 0 && @info "Done!" now(UTC) model.clock
MPI.Finalize()
