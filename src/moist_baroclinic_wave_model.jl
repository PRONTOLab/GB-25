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
using Breeze: AtmosphereModel, CompressibleDynamics, ExplicitTimeStepping
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density, specific_prognostic_moisture
using Breeze.Microphysics: NonEquilibriumCloudFormation
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: λnode, φnode, znode
using CloudMicrophysics
using SpecialFunctions
using CUDA
using Reactant

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

# ═══════════════════════════════════════════════════════════════════════════
# Model constructor
# ═══════════════════════════════════════════════════════════════════════════

"""
    moist_baroclinic_wave_model(arch; Nλ=360, Nφ=170, Nz=30, H=30e3, Δt=2.0, halo=(5,5,5))

Build a Breeze `AtmosphereModel` on a global LatitudeLongitudeGrid initialised
with the DCMIP-2016 moist baroclinic wave (Test 1-1).

Components
  • `CompressibleDynamics` with reference θ from the equatorial profile
  • `HydrostaticSphericalCoriolis`
  • `WENO(order=5)` advection (flux form)
  • `OneMomentCloudMicrophysics` (mixed-phase non-equilibrium with ice)

Grid: 0°–360° longitude, ±85° latitude, 0–`H` m altitude.
Default resolution ≈ 1° (Nλ=360, Nφ=170).
"""
function moist_baroclinic_wave_model(arch;
                                     Nλ   = 360,
                                     Nφ   = 170,
                                     Nz   = 30,
                                     H    = 30e3,
                                     Δt   = 2.0,
                                     halo = (5, 5, 5))

    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nλ, Nφ, Nz),
                                 halo,
                                 longitude = (0, 360),
                                 latitude  = (-85, 85),
                                 z = (0, H))

    coriolis = SphericalCoriolis()

    dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                   surface_pressure = p_ref,
                                   reference_potential_temperature = theta_reference)

    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    microphysics = ext.OneMomentCloudMicrophysics(;
        cloud_formation = NonEquilibriumCloudFormation(nothing, :ice))

    advection = WENO(order = 5)

    # Prescribed-SST bulk surface flux boundary conditions
    Cᴰ = 1e-3   # constant bulk exchange coefficient
    Uᵍ = 1e-2   # gustiness [m/s]
    T₀ = surface_temperature

    ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

    boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)

    model = AtmosphereModel(grid; dynamics, coriolis, advection, microphysics, boundary_conditions)

    model.clock.last_Δt = Δt

    set_moist_baroclinic_wave!(model)

    return model
end
