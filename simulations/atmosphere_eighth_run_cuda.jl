# Standalone vanilla-Oceananigans 1/8° atmosphere run with file-based IC.
#
# IMPORTANT: This script intentionally does NOT load `GordonBell25` or
# `Reactant`. On this box (driver 570 + Reactant pulling its own CUDA lib
# stack into the process) loading Reactant corrupts CUDA.jl's PTX target,
# so the very first Oceananigans GPU kernel — `gpu_compute_Δx_Az!` inside
# `LatitudeLongitudeGrid` precomputed-metrics — fails with
#   CUDA error: device kernel image is invalid (code 200, ERROR_INVALID_IMAGE).
# A bare `using Breeze, Oceananigans, CUDA` driver works fine, so we inline
# the bits of `src/moist_baroclinic_wave_model.jl` we need (constants,
# `theta_reference`, `surface_temperature`, the AtmosphereModel constructor,
# and the non-Reactant branch of the file-based IC loader) here.

using Dates
@info "This is when the fun begins" now(UTC)

using JLD2
using Printf
using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using CUDA
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics, ExplicitTimeStepping
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density
using Breeze.Microphysics: NonEquilibriumCloudFormation, ConstantRateCondensateFormation
using CloudMicrophysics  # triggers BreezeCloudMicrophysicsExt extension load

Oceananigans.defaults.FloatType = Float32

# ═══════════════════════════════════════════════════════════════════════════
# DCMIP-2016 balanced state (copied verbatim from src/moist_baroclinic_wave_model.jl)
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
# Model construction (mirrors GordonBell25.moist_baroclinic_wave_model
# minus the Reactant branches and minus `set_moist_baroclinic_wave!`)
# ═══════════════════════════════════════════════════════════════════════════

function build_atmosphere_model(arch; Nλ, Nφ, Nz, H, Δt, halo,
                                cloud_formation_τ_relax::Union{Nothing,Real} = nothing)
    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nλ, Nφ, Nz),
                                 halo,
                                 longitude = (0, 360),
                                 latitude  = (-80, 80),
                                 z = (0, H))

    coriolis = SphericalCoriolis()

    dynamics = CompressibleDynamics(
        ExplicitTimeStepping();
        surface_pressure = p_ref,
        reference_potential_temperature = theta_reference,
    )

    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    cloud_formation = if cloud_formation_τ_relax === nothing
        NonEquilibriumCloudFormation(nothing, nothing)
    else
        FT = Oceananigans.defaults.FloatType
        rate = FT(1) / FT(cloud_formation_τ_relax)
        cf = ConstantRateCondensateFormation{FT}(rate)
        NonEquilibriumCloudFormation(cf, cf)
    end
    microphysics = ext.OneMomentCloudMicrophysics(; cloud_formation)

    # WENO 5 on everything; positivity-bounded variant for moisture/cloud/precip.
    # An earlier experiment with WENO(3) on the microphysics tracers + Δt=1.0 NaN'd
    # within 256 steps; we reverted to WENO 5 on advice from the user (the Δt
    # change was the suspected culprit, not the order).
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
    T₀ = surface_temperature

    ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

    boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)

    model = AtmosphereModel(grid; dynamics, coriolis, momentum_advection,
                            scalar_advection, microphysics, boundary_conditions)

    FT = eltype(grid)
    model.clock.last_Δt = FT(Δt)

    return model
end

# ═══════════════════════════════════════════════════════════════════════════
# File-based IC loader — vanilla-Oceananigans branch only.
# Mirrors `set_moist_baroclinic_wave_from_file!` (src/moist_baroclinic_wave_model.jl:463).
# ═══════════════════════════════════════════════════════════════════════════

function load_ic_from_file!(model, path; H = 30e3)
    Nλ_src, Nφ_src, Nz_src, ρ_data, ρu_data, ρv_data, ρw_data, ρθ_data, ρqv_data =
        JLD2.jldopen(path, "r") do file
            (file["Nλ"], file["Nφ"], file["Nz"],
             file["ρ"], file["ρu"], file["ρv"], file["ρw"],
             file["ρθ"], file["ρqᵛ"])
        end

    expected_c  = (Nλ_src, Nφ_src,     Nz_src    )
    expected_yf = (Nλ_src, Nφ_src + 1, Nz_src    )
    expected_zf = (Nλ_src, Nφ_src,     Nz_src + 1)
    size(ρ_data)   == expected_c  || error("ρ size $(size(ρ_data)) ≠ $expected_c")
    size(ρu_data)  == expected_c  || error("ρu size $(size(ρu_data)) ≠ $expected_c")
    size(ρv_data)  == expected_yf || error("ρv size $(size(ρv_data)) ≠ $expected_yf")
    size(ρw_data)  == expected_zf || error("ρw size $(size(ρw_data)) ≠ $expected_zf")
    size(ρθ_data)  == expected_c  || error("ρθ size $(size(ρθ_data)) ≠ $expected_c")
    size(ρqv_data) == expected_c  || error("ρqᵛ size $(size(ρqv_data)) ≠ $expected_c")

    model_arch = model.grid.architecture
    source_grid = LatitudeLongitudeGrid(model_arch;
        size = (Nλ_src, Nφ_src, Nz_src),
        halo = (1, 1, 1),
        z = (0, H),
        latitude = (-80, 80),
        longitude = (0, 360),
    )

    ρ_src   = CenterField(source_grid)
    ρu_src  = XFaceField(source_grid)
    ρv_src  = YFaceField(source_grid)
    ρw_src  = ZFaceField(source_grid)
    ρθ_src  = CenterField(source_grid)
    ρqv_src = CenterField(source_grid)

    set!(ρ_src,   ρ_data)
    set!(ρu_src,  ρu_data)
    set!(ρv_src,  ρv_data)
    set!(ρw_src,  ρw_data)
    set!(ρθ_src,  ρθ_data)
    set!(ρqv_src, ρqv_data)

    for f in (ρ_src, ρu_src, ρv_src, ρw_src, ρθ_src, ρqv_src)
        Oceananigans.BoundaryConditions.fill_halo_regions!(f)
    end

    ρ_target   = dynamics_density(model.dynamics)
    ρu_target  = model.momentum.ρu
    ρv_target  = model.momentum.ρv
    ρw_target  = model.momentum.ρw
    ρθ_target  = model.formulation.potential_temperature_density
    ρqv_target = model.moisture_density

    Oceananigans.Fields.interpolate!(ρ_target,   ρ_src)
    Oceananigans.Fields.interpolate!(ρu_target,  ρu_src)
    Oceananigans.Fields.interpolate!(ρv_target,  ρv_src)
    Oceananigans.Fields.interpolate!(ρw_target,  ρw_src)
    Oceananigans.Fields.interpolate!(ρθ_target,  ρθ_src)
    Oceananigans.Fields.interpolate!(ρqv_target, ρqv_src)

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

@info "GPU info" CUDA.devices() CUDA.runtime_version()

arch = GPU(CUDABackend())

H_halo = 8
Nλ = 2880
Nφ = 1280
Nz = 64
column_height = 30e3   # m

# Acoustic CFL (vertical) is binding: Δt < Δz/c_s ≈ (30km/64)/340 m/s ≈ 1.38 s.
# An earlier test at Δt=1.0 NaN'd within 256 steps even though the acoustic
# CFL was satisfied — there's a tighter constraint we haven't identified
# (probably vertical advection at the top of the column where ρ → 0). Sticking
# with the known-good 0.5 s.
Δt = 0.5

# Cloud condensate formation relaxation timescale (s). Slower than the default
# instantaneous saturation adjustment.
cloud_τ_relax = 100.0

# Number of time steps and progress/checkpoint cadence.
total_steps   = 7200          # = 1 simulated hour at Δt = 0.5 s
log_every     = 360           # log every 3 simulated minutes
ckpt_every    = 1800          # save an intermediate checkpoint every 15 sim min

ic_path = joinpath(@__DIR__, "initial_conditions",
                   "atmosphere_eighth_explicit_microphysics_1hr.jld2")
isfile(ic_path) || error("IC file not found at $ic_path")
@info "Initial-condition file" ic_path

@info "Building atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, Δt=$(round(Δt; sigdigits=3))s, cloud_τ_relax=$(cloud_τ_relax)s)..." now(UTC)
@time "build model" model = build_atmosphere_model(arch; Nλ, Nφ, Nz, H=column_height, Δt,
                                                   halo=(H_halo, H_halo, H_halo),
                                                   cloud_formation_τ_relax=cloud_τ_relax)
@show model

@info "Loading initial conditions from file..." now(UTC)
@time "load ICs" load_ic_from_file!(model, ic_path; H=column_height)

function save_state!(model, path)
    ρ_field   = dynamics_density(model.dynamics)
    ρu_field  = model.momentum.ρu
    ρv_field  = model.momentum.ρv
    ρw_field  = model.momentum.ρw
    ρθ_field  = model.formulation.potential_temperature_density
    ρqv_field = model.moisture_density
    JLD2.jldsave(path;
                 ρ   = Array(interior(ρ_field)),
                 ρu  = Array(interior(ρu_field)),
                 ρv  = Array(interior(ρv_field)),
                 ρw  = Array(interior(ρw_field)),
                 ρθ  = Array(interior(ρθ_field)),
                 ρqᵛ = Array(interior(ρqv_field)),
                 Nλ  = size(model.grid, 1),
                 Nφ  = size(model.grid, 2),
                 Nz  = size(model.grid, 3),
                 time      = model.clock.time,
                 iteration = model.clock.iteration,
                 last_Δt   = model.clock.last_Δt)
    return nothing
end

# Mirror GordonBell25.first_time_step!(::AtmosphereModel) (src/timestepping_utils.jl:44):
# explicit update_state! then a single time_step!(model, Δt).
@info "First time step..." now(UTC)
@time "first time step" begin
    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.TimeSteppers.time_step!(model, Δt)
end

checkpoint_dir = joinpath(@__DIR__, "checkpoints")
mkpath(checkpoint_dir)
run_jobid = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH-MM-SS")

@info "Running $total_steps time steps (= $(total_steps*Δt) s simulated)..." now(UTC)
@time "total loop" begin
    for step in 1:total_steps
        Oceananigans.TimeSteppers.time_step!(model, Δt)

        if step % log_every == 0
            @info @sprintf("step %5d / %d  (sim t = %8.1f s = %5.2f h)",
                           step, total_steps, model.clock.time, model.clock.time / 3600) now(UTC)
            flush(stdout); flush(stderr)
        end

        if step % ckpt_every == 0 && step != total_steps
            ckpt_path = joinpath(checkpoint_dir,
                "atmosphere_eighth_run_cuda_$(run_jobid)_iter$(lpad(step, 6, '0')).jld2")
            @info "Intermediate checkpoint" ckpt_path
            @time "ckpt save" save_state!(model, ckpt_path)
            @info "Saved" ckpt_path filesize(ckpt_path)
            flush(stdout); flush(stderr)
        end
    end
end

@info "Done stepping" now(UTC) model.clock

final_path = joinpath(checkpoint_dir, "atmosphere_eighth_run_cuda_$(run_jobid)_final.jld2")
@info "Saving final checkpoint" final_path
@time "final ckpt save" save_state!(model, final_path)
@info "Saved" final_path filesize(final_path)

@info "Done!" now(UTC)
