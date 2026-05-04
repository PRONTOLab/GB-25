# Standalone vanilla-Oceananigans 1/8° atmosphere run, loaded from the 1° IC.
#
# Purpose: exercise THIS branch's nearest-neighbor IC loader
# (`set_moist_baroclinic_wave_from_file_vanilla!` + `_nn_atmos_field_copy!`
# in `src/moist_baroclinic_wave_model.jl`) under cross-resolution loading,
# to make sure it doesn't have a bug. The test does an 8× upsample
# (360×160×64 → 2880×1280×64), runs a few hundred time steps, and saves
# a checkpoint we can scan for NaN.
#
# IMPORTANT: This script intentionally does NOT load `GordonBell25` or
# `Reactant`. On this box loading Reactant corrupts CUDA.jl's PTX target,
# breaking the very first Oceananigans GPU kernel JIT (ERROR_INVALID_IMAGE).
# The relevant inlined pieces from `src/moist_baroclinic_wave_model.jl`:
#   - DCMIP-2016 balanced-state constants & helpers (theta_reference, surface_temperature)
#   - The model constructor (without the Reactant set! branch)
#   - The vanilla file-based IC loader and its `_nn_atmos_field_copy!` kernel

using Dates
@info "This is when the fun begins" now(UTC)
flush(stdout); flush(stderr)

using JLD2
using Printf
using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using KernelAbstractions: @kernel, @index
using CUDA
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics, ExplicitTimeStepping
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics  # triggers BreezeCloudMicrophysicsExt extension load

Oceananigans.defaults.FloatType = Float32

# ═══════════════════════════════════════════════════════════════════════════
# DCMIP-2016 balanced state (copied from src/moist_baroclinic_wave_model.jl)
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
# Inlined nearest-neighbor interpolation kernel (from this branch's
# src/moist_baroclinic_wave_model.jl:299).  Copies a (Nx_src×Ny_src×Nz_src)
# host-uploaded source array onto the (Nx_dst×Ny_dst×Nz_dst) interior of a
# device field via index-space rounding.  No source grid required.
# ═══════════════════════════════════════════════════════════════════════════

@kernel function _nn_atmos_field_copy!(field, src_data, Nx_src, Ny_src, Nz_src, Nx_dst, Ny_dst, Nz_dst)
    i, j, k = @index(Global, NTuple)
    i′ = clamp(ceil(Int, i * Nx_src / Nx_dst), 1, Nx_src)
    j′ = clamp(ceil(Int, j * Ny_src / Ny_dst), 1, Ny_src)
    k′ = clamp(ceil(Int, k * Nz_src / Nz_dst), 1, Nz_src)
    @inbounds field[i, j, k] = src_data[i′, j′, k′]
end

# ═══════════════════════════════════════════════════════════════════════════
# Inlined vanilla file-based IC loader (from this branch's
# src/moist_baroclinic_wave_model.jl:540, non-Reactant branch).
# ═══════════════════════════════════════════════════════════════════════════

function load_ic_from_file_vanilla!(model, path::String; H = 30e3)
    Nλ_src, Nφ_src, Nz_src, ρ_data, ρu_data, ρv_data, ρw_data, ρθ_data, ρqv_data =
        JLD2.jldopen(path, "r") do file
            (file["Nλ"], file["Nφ"], file["Nz"],
             file["ρ"], file["ρu"], file["ρv"], file["ρw"],
             file["ρθ"], file["ρqᵛ"])
        end

    ρqcl_data, ρqci_data = JLD2.jldopen(path, "r") do file
        ρqcl = haskey(file, "micro_ρqᶜˡ") ? file["micro_ρqᶜˡ"] : nothing
        ρqci = haskey(file, "micro_ρqᶜⁱ") ? file["micro_ρqᶜⁱ"] : nothing
        (ρqcl, ρqci)
    end

    expected_c  = (Nλ_src, Nφ_src,     Nz_src    )
    expected_xf = (Nλ_src, Nφ_src,     Nz_src    )
    expected_yf = (Nλ_src, Nφ_src + 1, Nz_src    )
    expected_zf = (Nλ_src, Nφ_src,     Nz_src + 1)
    size(ρ_data)   == expected_c  || error("ρ size $(size(ρ_data)) ≠ $expected_c from $path")
    size(ρu_data)  == expected_xf || error("ρu size $(size(ρu_data)) ≠ $expected_xf from $path")
    size(ρv_data)  == expected_yf || error("ρv size $(size(ρv_data)) ≠ $expected_yf from $path")
    size(ρw_data)  == expected_zf || error("ρw size $(size(ρw_data)) ≠ $expected_zf from $path")
    size(ρθ_data)  == expected_c  || error("ρθ size $(size(ρθ_data)) ≠ $expected_c from $path")
    size(ρqv_data) == expected_c  || error("ρqᵛ size $(size(ρqv_data)) ≠ $expected_c from $path")

    grid = model.grid
    arch = Oceananigans.architecture(grid)
    FT   = eltype(grid)

    pairs = Any[
        (FT.(ρ_data),   dynamics_density(model.dynamics)),
        (FT.(ρu_data),  model.momentum.ρu),
        (FT.(ρv_data),  model.momentum.ρv),
        (FT.(ρw_data),  model.momentum.ρw),
        (FT.(ρθ_data),  model.formulation.potential_temperature_density),
        (FT.(ρqv_data), model.moisture_density),
    ]

    if ρqcl_data !== nothing
        push!(pairs, (FT.(ρqcl_data), model.microphysical_fields[:ρqᶜˡ]))
        @info "Loading micro_ρqᶜˡ from IC file" extrema=extrema(ρqcl_data)
    end
    if ρqci_data !== nothing
        push!(pairs, (FT.(ρqci_data), model.microphysical_fields[:ρqᶜⁱ]))
        @info "Loading micro_ρqᶜⁱ from IC file" extrema=extrema(ρqci_data)
    end

    for (src_array, target_field) in pairs
        Nx_src_f, Ny_src_f, Nz_src_f = size(src_array)
        Nx_dst, Ny_dst, Nz_dst = size(Oceananigans.interior(target_field))

        # Move host source to the target architecture so the kernel can read it
        # from device code (no-op on CPU; host→device copy on GPU). Mirrors the
        # fix in src/moist_baroclinic_wave_model.jl:592.
        src_dev = Oceananigans.on_architecture(arch, src_array)

        @info "NN interpolate" field=nameof(typeof(target_field)) src=size(src_array) dst=(Nx_dst, Ny_dst, Nz_dst)
        Oceananigans.Utils.launch!(arch, grid, :xyz,
            _nn_atmos_field_copy!, target_field, src_dev,
            Nx_src_f, Ny_src_f, Nz_src_f, Nx_dst, Ny_dst, Nz_dst)

        Oceananigans.BoundaryConditions.fill_halo_regions!(target_field)
    end

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Model construction (mirrors GordonBell25.moist_baroclinic_wave_model
# minus the Reactant branches and minus `set_moist_baroclinic_wave!`).
# Cloud τ_relax is intentionally NOT set — the previous run with
# τ_relax=100 NaN'd; this run uses the default instantaneous adjustment.
# ═══════════════════════════════════════════════════════════════════════════

function build_atmosphere_model(arch; Nλ, Nφ, Nz, H, Δt, halo)
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
    cloud_formation = NonEquilibriumCloudFormation(nothing, nothing)
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
# State save (matches atmosphere_spinup_cuda.jl JLD2 schema so the result
# can be loaded back as IC).
# ═══════════════════════════════════════════════════════════════════════════

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

# Cheap GPU-side NaN check on the model's prognostic fields. Allocates
# a single Bool per field, doesn't transfer the whole field to host.
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

@info "GPU info" CUDA.devices() CUDA.runtime_version()
flush(stdout); flush(stderr)

arch = GPU(CUDABackend())

H_halo = 8
Nλ = 2880
Nφ = 1280
Nz = 64
column_height = 30e3   # m

# Acoustic CFL (vertical) is binding: Δt < (30km/64)/340 m/s ≈ 1.38 s.
Δt = 0.5

ic_path = joinpath(@__DIR__, "initial_conditions",
                   "atmosphere_no_microphysics_1deg_14day.jld2")
isfile(ic_path) || error("IC file not found at $ic_path")
@info "Initial-condition file (1° → 1/8° upsample, NN)" ic_path
flush(stdout); flush(stderr)

@info "Building atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, Δt=$(round(Δt; sigdigits=3))s)..." now(UTC)
flush(stdout); flush(stderr)
@time "build model" model = build_atmosphere_model(arch; Nλ, Nφ, Nz, H=column_height, Δt,
                                                   halo=(H_halo, H_halo, H_halo))
@show model
flush(stdout); flush(stderr)

@info "Loading initial conditions from 1° file via _nn_atmos_field_copy!..." now(UTC)
flush(stdout); flush(stderr)
@time "load ICs" load_ic_from_file_vanilla!(model, ic_path; H=column_height)
flush(stdout); flush(stderr)

if any_nan(model)
    error("Model state contains NaN immediately after IC load — interpolation bug")
end
@info "Post-load: no NaN in any prognostic field"
flush(stdout); flush(stderr)

# Mirror GordonBell25.first_time_step!(::AtmosphereModel) (src/timestepping_utils.jl:44):
# explicit update_state! then a single time_step!(model, Δt).
@info "First time step..." now(UTC)
flush(stdout); flush(stderr)
@time "first time step" begin
    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.TimeSteppers.time_step!(model, Δt)
end

if any_nan(model)
    error("NaN after first time step — interpolation may have produced invalid state")
end
@info "Post-first-step: no NaN" model.clock
flush(stdout); flush(stderr)

# Short probe loop with frequent NaN checks. We're testing IC loading correctness,
# not running a long simulation.
total_steps = 512
log_every   = 64
@info "Running $total_steps time steps (= $(total_steps*Δt) s simulated), NaN check every $log_every steps..." now(UTC)
flush(stdout); flush(stderr)
@time "loop $total_steps" begin
    for step in 1:total_steps
        Oceananigans.TimeSteppers.time_step!(model, Δt)
        if step % log_every == 0
            nan_now = any_nan(model)
            @info @sprintf("step %4d / %d  (sim t = %7.1f s)  nan=%s",
                           step, total_steps, model.clock.time, nan_now) now(UTC)
            flush(stdout); flush(stderr)
            nan_now && error("NaN detected at step $step")
        end
    end
end

@info "Done stepping" now(UTC) model.clock
flush(stdout); flush(stderr)

checkpoint_dir = joinpath(@__DIR__, "checkpoints")
mkpath(checkpoint_dir)
jobid = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH-MM-SS")
checkpoint_path = joinpath(checkpoint_dir, "atmosphere_eighth_from_1deg_$(jobid).jld2")

@info "Saving checkpoint" checkpoint_path
flush(stdout); flush(stderr)
@time "checkpoint save" save_state!(model, checkpoint_path)
@info "Saved" checkpoint_path filesize(checkpoint_path)

@info "Done!" now(UTC)
flush(stdout); flush(stderr)
