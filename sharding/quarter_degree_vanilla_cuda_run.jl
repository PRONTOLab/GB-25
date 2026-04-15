# Vanilla-Oceananigans 1/4° atmosphere run on a single CUDA GPU.
# Latitude range (-70, 70), Δt=2s, loaded from quarter-degree IC file.
#
# IMPORTANT: Does NOT load GordonBell25 or Reactant — on this box loading
# Reactant corrupts CUDA.jl's PTX target.

using Dates
@info "Starting vanilla CUDA quarter-degree atmosphere simulation" now(UTC)
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
using Breeze.AtmosphereModels: dynamics_density, specific_prognostic_moisture
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics

Oceananigans.defaults.FloatType = Float32

# ═══════════════════════════════════════════════════════════════════════════
# DCMIP-2016 balanced state (from src/moist_baroclinic_wave_model.jl)
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
# Vanilla file-based IC loader (interpolate! path)
# ═══════════════════════════════════════════════════════════════════════════

function load_ic_from_file!(model, path::String; H = 30e3)
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
        @info "Loading micro_ρqᶜˡ from IC file" extrema=extrema(ρqcl_data)
    end
    if ρqci_data !== nothing
        push!(pairs, (ρqci_data, model.microphysical_fields[:ρqᶜⁱ]))
        @info "Loading micro_ρqᶜⁱ from IC file" extrema=extrema(ρqci_data)
    end

    halo = Oceananigans.halo_size(grid)

    # Build source grid on the same architecture as the model
    src_grid = LatitudeLongitudeGrid(arch;
        size = (Nλ_src, Nφ_src, Nz_src),
        halo = halo,
        latitude  = (-80, 80),
        longitude = (-180, 180),
        z = (0, H))

    for (src_array, target_field) in pairs
        loc = Oceananigans.location(target_field)
        iloc = map(L -> L(), loc)
        src_field = Field(iloc, src_grid)
        gpu_data = Oceananigans.on_architecture(arch, Array{FT}(src_array))
        copyto!(Oceananigans.interior(src_field), gpu_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(src_field)

        @info "interpolate!" field=nameof(typeof(target_field)) loc src=size(Oceananigans.interior(src_field)) dst=size(Oceananigans.interior(target_field))
        Oceananigans.Fields.interpolate!(target_field, src_field)
    end

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Model construction
# ═══════════════════════════════════════════════════════════════════════════

function build_atmosphere_model(arch; Nλ, Nφ, Nz, H, Δt, halo, latitude)
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
    rate = FT(1) / FT(30)
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
    T₀ = surface_temperature

    ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

    boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)

    model = AtmosphereModel(grid; dynamics, coriolis, momentum_advection,
                            scalar_advection, microphysics, boundary_conditions)

    FT_grid = eltype(grid)
    model.clock.last_Δt = FT_grid(Δt)

    return model
end

# GPU-side NaN check
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

function report_state(model, label)
    fields = Oceananigans.fields(model)
    for name in keys(fields)
        f = fields[name]
        data = Array(Oceananigans.interior(f))
        mx = Float64(maximum(data))
        mn = Float64(minimum(data))
        @printf("  [%s] %6s: min=% .6e  max=% .6e\n", label, name, mn, mx)
        if isnan(mx) || isnan(mn)
            @error "NaN detected in $name at $label — aborting"
            exit(1)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

@info "GPU info" CUDA.devices() CUDA.runtime_version()
flush(stdout); flush(stderr)

arch = GPU()

Nλ = 1440
Nφ = 560
Nz = 64
column_height = 30e3
Δt = 2.0

ic_path = joinpath(@__DIR__, "..", "simulations", "initial_conditions",
                   "quarter_deg_day1_cloud_tau30.jld2")
isfile(ic_path) || error("IC file not found at $ic_path")
@info "IC file" ic_path
flush(stdout); flush(stderr)

@info "Building model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, latitude=(-70,70), Δt=$(Δt)s)..." now(UTC)
flush(stdout); flush(stderr)
@time "build model" model = build_atmosphere_model(arch; Nλ, Nφ, Nz,
                                                    H=column_height, Δt,
                                                    halo=(4, 4, 4),
                                                    latitude=(-70, 70))
@show model
flush(stdout); flush(stderr)

@info "Loading ICs..." now(UTC)
flush(stdout); flush(stderr)
@time "load ICs" load_ic_from_file!(model, ic_path; H=column_height)
flush(stdout); flush(stderr)

if any_nan(model)
    error("NaN after IC load")
end
@info "Post-load: no NaN"

@info "Initial state:"
report_state(model, "IC")

# ── First time step ────────────────────────────────────────────────────

@info "First time step..." now(UTC)
flush(stdout); flush(stderr)
@time "first step" begin
    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.TimeSteppers.time_step!(model, Δt)
end

if any_nan(model)
    error("NaN after first time step")
end
report_state(model, "step 1")

# ── Main loop (~1 year) ───────────────────────────────────────────────

Ninner = 1024
Nouter = 15410  # ~1 year with Δt=2s

@info "Starting main loop: $Nouter blocks × $Ninner steps (Δt=$Δt)" now(UTC)
flush(stdout); flush(stderr)

wall_start = time_ns()
for k in 1:Nouter
    t0 = time_ns()
    for _ in 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    wall_block = (time_ns() - t0) / 1e9
    total_steps = Ninner * k + 1
    sim_time = total_steps * Δt
    total_wall = (time_ns() - wall_start) / 1e9
    sim_days = sim_time / 86400
    sypd = (Ninner * Δt) / (365.25 * 86400 * wall_block) * 365.25

    @info @sprintf("block %d/%d: %d steps, wall=%.1fs, sim_time=%.0fs (day %.1f), SYPD=%.4f, total_wall=%.0fs",
                    k, Nouter, total_steps, wall_block, sim_time, sim_days, sypd, total_wall)

    if any_nan(model)
        @error "NaN detected at block $k — aborting"
        exit(1)
    end

    flush(stderr); flush(stdout)
end

@info "Done!" now(UTC)
