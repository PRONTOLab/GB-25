#####
##### DCMIP-2016 Moist Baroclinic Wave — Test 1-1
#####
# Reference: Ullrich, Jablonowski, Reed, Zarzycki, Lauritzen, Nair, Kent, Verlet-Banide (2016)
#   "Dynamical Core Model Intercomparison Project (DCMIP) Test Case Document"
# Fortran: github.com/ClimateGlobalChange/DCMIP2016/interface/baroclinic_wave_test.f90
#
# The balanced state (Ullrich et al. 2014) is expressed in terms of virtual
# temperature Tv(φ, z).  For the moist case the actual temperature is
# T = Tv / (1 + ε·q), so the moist atmosphere is colder than the dry one
# while maintaining the same pressure-gradient forcing.
# Density always uses virtual temperature:  ρ = p / (Rd · Tv).

using KernelAbstractions: @kernel, @index
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics, ExplicitTimeStepping,
              SplitExplicitTimeDiscretization, PressureProjectionDamping
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density, specific_prognostic_moisture
using Breeze.Microphysics: NonEquilibriumCloudFormation, BulkMicrophysics
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Grids: λnode, φnode, znode, Center, Face, LatitudeLongitudeGrid
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField, Field
using CloudMicrophysics
using SpecialFunctions
using CUDA
using Reactant
using Reactant: @jit, InterpolateArray, InterpolationType
using JLD2

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════

const earth_radius   = 6371220.0      # [m]
const earth_rotation = 7.29212e-5     # [s⁻¹]
const gravity        = 9.80616        # [m s⁻²]
const Rd_dry         = 287.0          # [J kg⁻¹ K⁻¹]  dry-air gas constant
const cp_dry         = 1004.5         # [J kg⁻¹ K⁻¹]  specific heat at constant pressure
const κ_exponent     = 2.0 / 7.0     # Rd / cp
const Rv_vapor       = 461.5          # [J kg⁻¹ K⁻¹]  water-vapor gas constant
const ε_virtual      = 0.608          # Rv/Rd − 1 ≈ 0.608  (virtual-temperature factor)
const p_ref          = 1e5            # [Pa]  reference surface pressure

# ═══════════════════════════════════════════════════════════════════════════
# Balanced-state parameters  (Table VI, Ullrich et al. 2014)
# ═══════════════════════════════════════════════════════════════════════════

const T_equator    = 310.0            # [K]   equatorial surface temperature
const T_polar      = 240.0            # [K]   polar surface temperature
const T_mean       = 0.5 * (T_equator + T_polar)   # 275 K
const lapse_rate   = 0.005            # [K m⁻¹]  lapse-rate parameter Γ
const jet_width    = 3.0              # K parameter  (jet width)
const vert_width   = 2.0              # b parameter  (vertical half-width)

# Derived coefficients for the τ-integrals  (Eqs. 5–8)
const coeff_A      = 1.0 / lapse_rate
const coeff_B      = (T_mean - T_polar) / (T_mean * T_polar)
const coeff_C      = 0.5 * (jet_width + 2) * (T_equator - T_polar) / (T_equator * T_polar)
const scale_height = Rd_dry * T_mean / gravity     # ≈ 8 km

# ═══════════════════════════════════════════════════════════════════════════
# Perturbation parameters  (exponential type, Eq. 14)
# ═══════════════════════════════════════════════════════════════════════════

const u_perturb  = 1.0               # [m s⁻¹]  amplitude
const r_perturb  = 0.1               # perturbation radius  [Earth radii]
const λ_center   = π / 9             # 20°E   [rad]
const φ_center   = 2π / 9            # 40°N   [rad]
const z_perturb  = 15000.0           # [m]    height cap

# ═══════════════════════════════════════════════════════════════════════════
# Moisture parameters  (DCMIP-2016, Eq. 18)
# ═══════════════════════════════════════════════════════════════════════════
#   q(λ,φ,η) = q₀ exp[−(φ/φ_w)⁴] exp[−((η−1)p₀/p_w)²]   for η > η_t
#   q         = q_t                                          above tropopause

const q0_surface   = 0.018            # [kg kg⁻¹]  peak specific humidity
const φ_width      = 2π / 9           # [rad]  40° — latitudinal e-folding width
const p_width      = 34000.0          # [Pa]   340 hPa — vertical pressure width
const η_tropopause = 0.1              # p/p_ref cutoff
const q_tropopause = 1e-12            # [kg kg⁻¹]  humidity above the tropopause

# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers  (all angles in radians)
# ═══════════════════════════════════════════════════════════════════════════

"""
    vertical_structure(z) → (τ₁, τ₂, I₁, I₂)

Vertical profiles and their integrals that define the balanced state (Eqs. 5–8).
"""
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

"""Horizontal temperature structure:  cos(φ)^K − K/(K+2) cos(φ)^(K+2)."""
F_temperature(cosφ) = cosφ^jet_width - jet_width / (jet_width + 2) * cosφ^(jet_width + 2)

"""Horizontal wind structure:  cos(φ)^(K−1) − cos(φ)^(K+1)."""
F_wind(cosφ) = cosφ^(jet_width - 1) - cosφ^(jet_width + 1)

"""
    virtual_temperature(φ, z)

Virtual temperature Tv from the balanced state (Eq. 9, shallow atmosphere).
In the dry case Tv = T; in the moist case Tv > T.
"""
function virtual_temperature(φ, z)
    vs = vertical_structure(z)
    return 1.0 / (vs.τ₁ - vs.τ₂ * F_temperature(cos(φ)))
end

"""
    balanced_pressure(φ, z)

Hydrostatic pressure from the balanced state (Eq. 10).
"""
function balanced_pressure(φ, z)
    vs = vertical_structure(z)
    return p_ref * exp(-gravity / Rd_dry * (vs.I₁ - vs.I₂ * F_temperature(cos(φ))))
end

"""
    moisture_profile(φ, z)

DCMIP-2016 specific humidity (Eq. 18).
Below the tropopause (η > 0.1): concentrated at midlatitudes in the lower troposphere,
peaking around 85 % relative humidity.  Above: q ≈ 10⁻¹² kg/kg.
"""
function moisture_profile(φ, z)
    p = balanced_pressure(φ, z)
    η = p / p_ref
    q_below = q0_surface * exp(-(φ / φ_width)^4) *
                           exp(-((η - 1) * p_ref / p_width)^2)
    return ifelse(η > η_tropopause, q_below, q_tropopause)
end

"""
    balanced_zonal_wind(φ, z)

Gradient-wind–balanced zonal wind (Eq. 12, shallow atmosphere).
"""
function balanced_zonal_wind(φ, z)
    vs   = vertical_structure(z)
    cosφ = cos(φ)
    Tv   = 1.0 / (vs.τ₁ - vs.τ₂ * F_temperature(cosφ))

    U_term   = gravity / earth_radius * jet_width * vs.I₂ * F_wind(cosφ) * Tv
    r_cosφ   = earth_radius * cosφ
    Ω_r_cosφ = earth_rotation * r_cosφ

    return -Ω_r_cosφ + sqrt(Ω_r_cosφ^2 + r_cosφ * U_term)
end

"""
    wind_perturbation(λ, φ, z)

Exponential perturbation to the zonal wind (Eq. 14).
Gaussian in great-circle distance from (λ_center, φ_center), tapered above z_perturb.
"""
function wind_perturbation(λ, φ, z)
    # Haversine great-circle distance squared — avoids acos/atan2 entirely.
    # sin²(d/2) = sin²(Δφ/2) + cos(φ)cos(φc)sin²(Δλ/2)
    # For small d: d² ≈ 4·sin²(d/2), exact to O(d⁴).
    sin_dφ = sin((φ - φ_center) / 2)
    sin_dλ = sin((λ - λ_center) / 2)
    h = sin_dφ * sin_dφ + cos(φ) * cos(φ_center) * sin_dλ * sin_dλ
    gc_sq = 4.0 * h / (r_perturb * r_perturb)

    ẑ     = z / z_perturb
    taper = ifelse(z < z_perturb, 1 - 3 * ẑ^2 + 2 * ẑ^3, 0.0)

    return ifelse(gc_sq < 1.0, u_perturb * taper * exp(-gc_sq), 0.0)
end

# ═══════════════════════════════════════════════════════════════════════════
# Public IC functions for  set!(model; θ, u, ρ, qᵛ)
#   Accepts (λ_deg, φ_deg, z) — LatitudeLongitudeGrid coordinates in degrees.
# ═══════════════════════════════════════════════════════════════════════════

"""
Actual potential temperature:  θ = T · (p_ref / p)^κ
where T = Tv / (1 + ε·q) is the actual (not virtual) temperature.
"""
function initial_theta(λ_deg, φ_deg, z)
    φ  = deg2rad(φ_deg)
    Tv = virtual_temperature(φ, z)
    p  = balanced_pressure(φ, z)
    q  = moisture_profile(φ, z)
    T  = Tv / (1 + ε_virtual * q)
    return T * (p_ref / p)^κ_exponent
end

"""Density  ρ = p / (Rd · Tv)  — uses virtual temperature (moist ideal-gas law)."""
function initial_density(λ_deg, φ_deg, z)
    φ = deg2rad(φ_deg)
    return balanced_pressure(φ, z) / (Rd_dry * virtual_temperature(φ, z))
end

"""Zonal wind: gradient-wind balance + exponential perturbation."""
function initial_zonal_wind(λ_deg, φ_deg, z)
    λ = deg2rad(λ_deg)
    φ = deg2rad(φ_deg)
    return balanced_zonal_wind(φ, z) + wind_perturbation(λ, φ, z)
end

"""Specific humidity from the DCMIP-2016 moist profile (Eq. 18)."""
initial_moisture(λ_deg, φ_deg, z) = moisture_profile(deg2rad(φ_deg), z)

"""Reference potential temperature at the equator for base-state subtraction."""
theta_reference(z) = initial_theta(0.0, 0.0, z)

"""
    surface_temperature(λ_deg, φ_deg)

Prescribed SST from the DCMIP-2016 balanced state evaluated at z = 0.
Returns virtual temperature at the surface, which for the bulk flux
formulas serves as the ocean skin temperature driving sensible/latent
heat exchange.
"""
surface_temperature(λ_deg, φ_deg) = virtual_temperature(deg2rad(φ_deg), 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# Initial-condition kernels
# ═══════════════════════════════════════════════════════════════════════════

@kernel function _set_moist_baroclinic_wave_kernel!(θ_field, ρ_field, qv_field, grid)
    i, j, k = @index(Global, NTuple)
    λ_deg = λnode(i, j, k, grid, Center(), Center(), Center())
    φ_deg = φnode(i, j, k, grid, Center(), Center(), Center())
    z     = znode(i, j, k, grid, Center(), Center(), Center())
    @inbounds begin
        θ_field[i, j, k] = initial_theta(λ_deg, φ_deg, z)
        ρ_field[i, j, k] = initial_density(λ_deg, φ_deg, z)
        qv_field[i, j, k] = initial_moisture(λ_deg, φ_deg, z)
    end
end

@kernel function _set_zonal_wind_kernel!(u_field, grid)
    i, j, k = @index(Global, NTuple)
    λ_deg = λnode(i, j, k, grid, Face(), Center(), Center())
    φ_deg = φnode(i, j, k, grid, Face(), Center(), Center())
    z     = znode(i, j, k, grid, Face(), Center(), Center())
    @inbounds u_field[i, j, k] = initial_zonal_wind(λ_deg, φ_deg, z)
end

function _set_moist_baroclinic_wave!(model)
    grid = model.grid
    arch = grid.architecture

    ρ  = dynamics_density(model.dynamics)
    θ  = model.formulation.potential_temperature
    qv = specific_prognostic_moisture(model)
    u  = model.velocities.u

    Oceananigans.Utils.launch!(arch, grid, :xyz,
        _set_moist_baroclinic_wave_kernel!, θ, ρ, qv, grid)

    Oceananigans.Utils.launch!(arch, grid, :xyz,
        _set_zonal_wind_kernel!, u, grid)

    ρθ = model.formulation.potential_temperature_density
    parent(ρθ) .= parent(ρ) .* parent(θ)

    ρu = model.momentum.ρu
    parent(ρu) .= parent(ρ) .* parent(u)

    ρqv = model.moisture_density
    parent(ρqv) .= parent(ρ) .* parent(qv)

    return nothing
end

function set_moist_baroclinic_wave!(model)
    if model.grid.architecture isa ReactantState
        rset! = @compile sync=true raise=true _set_moist_baroclinic_wave!(model)
        rset!(model)
    else
        _set_moist_baroclinic_wave!(model)
    end
end

@kernel function _nn_atmos_field_copy!(field, src_data, Nx_src, Ny_src, Nz_src, Nx_dst, Ny_dst, Nz_dst)
    i, j, k = @index(Global, NTuple)
    i′ = clamp(ceil(Int, i * Nx_src / Nx_dst), 1, Nx_src)
    j′ = clamp(ceil(Int, j * Ny_src / Ny_dst), 1, Ny_src)
    k′ = clamp(ceil(Int, k * Nz_src / Nz_dst), 1, Nz_src)
    @inbounds field[i, j, k] = src_data[i′, j′, k′]
end

# ═══════════════════════════════════════════════════════════════════════════
# Model constructor
# ═══════════════════════════════════════════════════════════════════════════

"""
    moist_baroclinic_wave_model(arch; Nλ=360, Nφ=160, Nz=64, H=30e3, Δt=nothing,
                                halo=(8,8,8), time_discretization=ExplicitTimeStepping(), ...)

Build a Breeze `AtmosphereModel` on a global LatitudeLongitudeGrid initialised
with the DCMIP-2016 moist baroclinic wave (Test 1-1).

Time discretization options (`time_discretization` keyword):
  • `ExplicitTimeStepping()` (default) — no acoustic substepping. Δt must
    satisfy the acoustic CFL (binding constraint is Δz / c_s ≈ 1.4 s for
    Nz=64, H=30 km).
  • `SplitExplicitTimeDiscretization(damping = PressureProjectionDamping(coefficient=0.5))` —
    WS-RK3 + acoustic substepping. The outer Δt follows the advective CFL.
    Use `coefficient=0.5` for the moist BCW (the Breeze 0.4.6 default of
    0.1 is too weak and silently produces NaN-filled state).

Other components:
  • `HydrostaticSphericalCoriolis`
  • `WENO(order=5)` advection (flux form)
  • `OneMomentCloudMicrophysics` (mixed-phase non-equilibrium with ice)

Grid: 0°–360° longitude, ±80° latitude, 0–`H` m altitude.
Default resolution ≈ 1° (Nλ=360, Nφ=160) with Nz=64 vertical levels and
halo=(8,8,8) so WENO(5) and the spinup/sharded scripts agree on halo width.

Time step: when `Δt === nothing`, a conservative formula is used:
  ExplicitTimeStepping:           `Δt = 0.5 · Δz / 330` s   (acoustic CFL).
  SplitExplicitTimeDiscretization: `Δt = 60 · (360 / Nλ)` s (advective CFL).
"""
function moist_baroclinic_wave_model(arch;
                                     Nλ           = 360,
                                     Nφ           = 160,
                                     Nz           = 64,
                                     H            = 30e3,
                                     Δt           = nothing,
                                     halo         = (8, 8, 8),
                                     time_discretization = ExplicitTimeStepping(),
                                     with_microphysics = true,
                                     with_surface_fluxes = true,
                                     with_advection = true,
                                     with_coriolis = true,
                                     sst_anomaly = 0,
                                     cloud_formation_τ_relax::Union{Nothing,Real} = nothing,
                                     initial_conditions_path::Union{Nothing,String} = nothing,
                                     interpolation_type = :nearest)

    if isnothing(Δt)
        if time_discretization isa SplitExplicitTimeDiscretization
            Δt_value = 60.0 * (360 / Nλ)
        else
            Δt_value = 0.5 * (H / Nz) / 330.0
        end
    else
        Δt_value = Δt
    end

    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nλ, Nφ, Nz),
                                 halo,
                                 longitude = (0, 360),
                                 latitude  = (-80, 80),
                                 z = (0, H))

    coriolis = with_coriolis ? SphericalCoriolis() : nothing

    dynamics = CompressibleDynamics(
        time_discretization;
        surface_pressure = p_ref,
        reference_potential_temperature = theta_reference,
    )

    microphysics = if with_microphysics isa Symbol
        # Named microphysics variants for ablation testing
        if with_microphysics === :warm_1m
            # Warm-phase 1M: liquid cloud formation only, no ice
            ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
            FT = Oceananigans.defaults.FloatType
            if cloud_formation_τ_relax === nothing
                cloud_formation = NonEquilibriumCloudFormation(nothing, nothing)  # defaults, but ice path inactive
            else
                rate = FT(1) / FT(cloud_formation_τ_relax)
                cf = Breeze.Microphysics.ConstantRateCondensateFormation{FT}(rate)
                cloud_formation = NonEquilibriumCloudFormation(cf, nothing)  # liquid only
            end
            ext.OneMomentCloudMicrophysics(; cloud_formation)
        elseif with_microphysics === :kessler
            # DCMIP2016 Kessler warm-rain scheme (no ice at all)
            Breeze.Microphysics.DCMIP2016KesslerMicrophysics()
        else
            error("Unknown microphysics type: $with_microphysics")
        end
    elseif with_microphysics
        ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
        cloud_formation = if cloud_formation_τ_relax === nothing
            NonEquilibriumCloudFormation(nothing, nothing)
        else
            FT = Oceananigans.defaults.FloatType
            rate = FT(1) / FT(cloud_formation_τ_relax)
            cf = Breeze.Microphysics.ConstantRateCondensateFormation{FT}(rate)
            NonEquilibriumCloudFormation(cf, cf)
        end
        ext.OneMomentCloudMicrophysics(; cloud_formation)
    else
        nothing
    end

    # Bounds-preserving WENO for moisture/cloud/precip tracers (rico.jl pattern):
    # plain WENO can produce small negatives in qᵛ et al, which then feed back
    # through microphysics tendencies and blow up. Clamp to [0, 1] at the
    # advection step.
    if with_advection
        weno     = WENO(order = 5)
        weno_pos = WENO(order = 5, bounds = (0, 1))
        momentum_advection = weno
        scalar_advection = if with_microphysics
            (ρθ   = weno,
             ρqᵛ  = weno_pos,
             ρqᶜˡ = weno_pos,
             ρqᶜⁱ = weno_pos,
             ρqʳ  = weno_pos,
             ρqˢ  = weno_pos)
        else
            weno
        end
    else
        momentum_advection = nothing
        scalar_advection = nothing
    end

    # Prescribed-SST bulk surface flux boundary conditions
    boundary_conditions = if with_surface_fluxes
        Cᴰ = 1e-3   # constant bulk exchange coefficient
        Uᵍ = 1e-2   # gustiness [m/s]
        T₀ = (λ, φ) -> surface_temperature(λ, φ) + sst_anomaly

        ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
        ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
        ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
        ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

        (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)
    else
        NamedTuple()
    end

    model = AtmosphereModel(grid; dynamics, coriolis, momentum_advection, scalar_advection,
                            microphysics, boundary_conditions)

    FT = eltype(grid)
    model.clock.last_Δt = FT(Δt_value)

    if initial_conditions_path === nothing
        set_moist_baroclinic_wave!(model)
    else
        grid_arch = model.grid.architecture
        if grid_arch isa ReactantState || grid_arch isa ShardedDistributedArch
            set_moist_baroclinic_wave_from_file!(model, initial_conditions_path; H = H, interpolation_type)
        else
            set_moist_baroclinic_wave_from_file_vanilla!(model, initial_conditions_path; H = H, interpolation_type)
        end
    end

    return model
end

# ═══════════════════════════════════════════════════════════════════════════
# Initial conditions from a JLD2 checkpoint
# ═══════════════════════════════════════════════════════════════════════════

"""
    set_moist_baroclinic_wave_from_file!(model, path::String; H = 30e3)

Load the prognostic state (`ρ, ρu, ρv, ρw, ρθ, ρqᵛ`) from a JLD2 checkpoint
and interpolate onto the model fields via `Reactant.InterpolateArray`.

For each field the target size and sharding are read directly from the model
field's backing `ConcreteIFRTArray`, so this works identically for single-device
and distributed (sharded) Reactant architectures — including face fields whose
per-rank parent shape matches center fields on distributed grids.
"""
function set_moist_baroclinic_wave_from_file!(model, path::String; H = 30e3, interpolation_type = :nearest)
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
    halo = Oceananigans.halo_size(grid)
    FT   = eltype(grid)

    pairs = [
        (ρ_data,   dynamics_density(model.dynamics)),
        (ρu_data,  model.momentum.ρu),
        (ρv_data,  model.momentum.ρv),
        (ρw_data,  model.momentum.ρw),
        (ρθ_data,  model.formulation.potential_temperature_density),
        (ρqv_data, model.moisture_density),
    ]

    has_microphysics = hasproperty(model, :microphysical_fields) && !isempty(model.microphysical_fields)
    if ρqcl_data !== nothing && has_microphysics && haskey(model.microphysical_fields, :ρqᶜˡ)
        push!(pairs, (ρqcl_data, model.microphysical_fields[:ρqᶜˡ]))
        @info "Loading micro_ρqᶜˡ from IC file" extrema=extrema(ρqcl_data)
    end
    if ρqci_data !== nothing && has_microphysics && haskey(model.microphysical_fields, :ρqᶜⁱ)
        push!(pairs, (ρqci_data, model.microphysical_fields[:ρqᶜⁱ]))
        @info "Loading micro_ρqᶜⁱ from IC file" extrema=extrema(ρqci_data)
    end

    # Build a temporary CPU grid at source resolution so we can create proper
    # fields with halos filled by Oceananigans (periodic in λ, bounded in φ/z).
    src_grid = LatitudeLongitudeGrid(CPU();
        size = (Nλ_src, Nφ_src, Nz_src),
        halo = halo,
        latitude  = (-80, 80),
        longitude = (-180, 180),
        z = (0, H))

    for (src_array, target_field) in pairs
        target_data = Reactant.ancestor(target_field)
        target_size = size(target_data)
        sharding    = target_data.sharding

        # 1. Create a source field on the CPU grid with matching location,
        #    set its interior from the JLD2 data, and fill halos properly.
        loc = Oceananigans.location(target_field)
        iloc = map(L -> L(), loc)  # instantiate: (Center, Face, Center) → (Center(), Face(), Center())
        src_field = Field(iloc, src_grid)
        Oceananigans.interior(src_field) .= FT.(src_array)
        Oceananigans.BoundaryConditions.fill_halo_regions!(src_field)
        src_parent = parent(src_field)

        # 2. Trim src_parent to Reactant's parent convention.
        #    Vanilla: total_length(Face, Bounded, N, H) = N + 1 + 2H
        #    Reactant: reactant_total_length(Face, Bounded, N, H) = N + 2H
        #    So we drop the last row in face+bounded dimensions.
        topo = Oceananigans.topology(src_grid)
        trim(d) = loc[d] === Face && topo[d] === Bounded
        rx = trim(1) ? (1:size(src_parent,1)-1) : Colon()
        ry = trim(2) ? (1:size(src_parent,2)-1) : Colon()
        rz = trim(3) ? (1:size(src_parent,3)-1) : Colon()
        src_reactant = view(src_parent, rx, ry, rz)

        @info "InterpolateArray" field=nameof(typeof(target_field)) loc src_parent=size(src_parent) src_reactant=size(src_reactant) dst=target_size halo

        itype = interpolation_type === :nearest ? InterpolationType.Nearest : InterpolationType.Linear
        result = InterpolateArray(src_reactant, target_size, sharding, itype, halo)
        target_data.data     = result.data
        target_data.sharding = result.sharding

        Oceananigans.BoundaryConditions.fill_halo_regions!(target_field)
    end

    # # Clamp ρqᵛ to be non-negative. The 1/8° source IC can contain small
    # # negative moisture values (min ~-1e-4) from bounds-preserving WENO that,
    # # when interpolated onto a finer grid and hit by the first microphysics
    # # step, reliably produce NaN in particular cells on particular tiles.
    # # Clipping here removes that root cause of iter-2 blowups in the fine-
    # # resolution spinup.
    # ρqv_target = model.moisture_density
    # qv_int = Oceananigans.interior(ρqv_target)
    # qv_int .= max.(qv_int, zero(FT))
    # Oceananigans.BoundaryConditions.fill_halo_regions!(ρqv_target)

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Initial conditions from a JLD2 checkpoint — vanilla Oceananigans
# ═══════════════════════════════════════════════════════════════════════════

"""
    set_moist_baroclinic_wave_from_file_vanilla!(model, path::String; H = 30e3)

Load the prognostic state (`ρ, ρu, ρv, ρw, ρθ, ρqᵛ` and optionally `ρqᶜˡ, ρqᶜⁱ`)
from a JLD2 checkpoint and interpolate onto the model fields using
Oceananigans' `interpolate!`.  Works on CPU (and GPU) without Reactant.
"""
function set_moist_baroclinic_wave_from_file_vanilla!(model, path::String; H = 30e3, halo = (4, 4, 4), interpolation_type = :nearest)
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

    pairs = [
        (ρ_data,   dynamics_density(model.dynamics)),
        (ρu_data,  model.momentum.ρu),
        (ρv_data,  model.momentum.ρv),
        (ρw_data,  model.momentum.ρw),
        (ρθ_data,  model.formulation.potential_temperature_density),
        (ρqv_data, model.moisture_density),
    ]

    has_micro = hasproperty(model, :microphysical_fields) && !isempty(model.microphysical_fields)
    if ρqcl_data !== nothing && has_micro && haskey(model.microphysical_fields, :ρqᶜˡ)
        push!(pairs, (FT.(ρqcl_data), model.microphysical_fields[:ρqᶜˡ]))
        @info "Loading micro_ρqᶜˡ from IC file" extrema=extrema(ρqcl_data)
    end
    if ρqci_data !== nothing && has_micro && haskey(model.microphysical_fields, :ρqᶜⁱ)
        push!(pairs, (FT.(ρqci_data), model.microphysical_fields[:ρqᶜⁱ]))
        @info "Loading micro_ρqᶜⁱ from IC file" extrema=extrema(ρqci_data)
    end

    src_grid = LatitudeLongitudeGrid(CPU();
        size = (Nλ_src, Nφ_src, Nz_src),
        halo = halo,
        latitude  = (-80, 80),
        longitude = (-180, 180),
        z = (0, H))

    for (src_array, target_field) in pairs
        loc = Oceananigans.location(target_field)
        iloc = map(L -> L(), loc)
        src_field = Field(iloc, src_grid)
        Oceananigans.interior(src_field) .= FT.(src_array)
        Oceananigans.BoundaryConditions.fill_halo_regions!(src_field)

        @info "interpolate!" field=nameof(typeof(target_field)) loc src=size(Oceananigans.interior(src_field)) dst=size(Oceananigans.interior(target_field))
        Oceananigans.Fields.interpolate!(target_field, src_field)
    end

    # Clamp ρqᵛ to be non-negative. The 1/8° source IC can contain small
    # negative moisture values (min ~-1e-4) from bounds-preserving WENO that,
    # when interpolated onto a finer grid and hit by the first microphysics
    # step, reliably produce NaN in particular cells on particular tiles.
    # Clipping here removes that root cause of iter-2 blowups in the fine-
    # resolution spinup.
    ρqv_target = model.moisture_density
    qv_int = Oceananigans.interior(ρqv_target)
    qv_int .= max.(qv_int, zero(FT))
    Oceananigans.BoundaryConditions.fill_halo_regions!(ρqv_target)

    return nothing
end

@static if isdefined(Core, :BFloat16)
    SpecialFunctions.gamma(x::Core.BFloat16) =
        Core.BFloat16(SpecialFunctions.gamma(Float32(x)))
end
