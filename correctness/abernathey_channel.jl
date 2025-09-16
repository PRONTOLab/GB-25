#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode, XDirection, YDirection, ZDirection, peripheral_node, static_column_depthᶜᶜᵃ, rnode, getnode, inactive_cell
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

using SeawaterPolynomials

#using Oceananigans.Architectures: GPU
#using CUDA
#CUDA.device!(0)

using Reactant
using GordonBell25
using Oceananigans.Architectures: ReactantState, architecture
#Reactant.set_default_backend("cpu")

using Enzyme


Oceananigans.defaults.FloatType = Float64

#
# Model parameters to set first:
#

# number of grid points
Nx = 8
Ny = 8
Nz = 8

# stretched grid
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = sum(Δz_center)

# Coriolis variables:
const f = -1e-4
const β = 1e-11

halo_size = 4 #3 for non-immersed grid

# Other model parameters:
const α = 2e-4     # [K⁻¹] thermal expansion coefficient
const g = 9.8061   # [m/s²] gravitational constant
const cᵖ = 3994.0   # [J/K]  heat capacity
const ρ = 999.8    # [kg/m³] reference density

parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,            # buoyancy flux magnitude [m² s⁻³]
    Qᵀ = 10 / (ρ * cᵖ),                    # temperature flux magnitude
    y_shutoff = 5 / 6 * Ly,                # shutoff location for buoyancy flux [m]
    τ = 0.2 / ρ,                           # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
    ΔT = 8,                              # surface vertical temperature gradient
    H = Lz,                              # domain depth [m]
    h = 1000.0,                          # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,               # southern boundary of sponge layer [m]
    λt = 7.0days                         # relaxation time scale [s]
)

# full ridge function:
function ridge_function(x, y)
    zonal = (Lz+100)exp(-(x - Lx/2)^2/(1e6kilometers))
    gap   = 1 - 0.5(tanh((y - (Ly/6))/1e5) - tanh((y - (Ly/2))/1e5))
    return zonal * gap - Lz
end

function make_grid(architecture, Nx, Ny, Nz, Δz_center)
    z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
    z_faces[Nz+1] = 0

    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    # Make into a ridge array:
    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, ridge_function)

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))
    return grid
end

#####
##### Model construction:
#####

function build_model(grid, Δt₀, parameters)

    @inline function temperature_flux(i, j, grid, clock, model_fields, p)
        y = ynode(j, grid, Center())
        return ifelse(y < p.y_shutoff, p.Qᵀ * cos(3π * y / p.Ly), 0.0)
    end

    temperature_flux_bc = FluxBoundaryCondition(temperature_flux, discrete_form = true, parameters = parameters)

    u_stress_bc = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    T_bcs = FieldBoundaryConditions(top = temperature_flux_bc)

    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

    #####
    ##### Coriolis
    #####
    coriolis = BetaPlane(f₀ = f, β = β)

    #####
    ##### Forcing and initial condition
    #####
    @inline initial_temperature(z, p) = p.ΔT * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
    @inline mask(y, p)                = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

    @inline function temperature_relaxation(i, j, k, grid, clock, model_fields, p)
        timescale = p.λt
        y = ynode(j, grid, Center())
        z = znode(k, grid, Center())
        target_T = initial_temperature(z, p)
        T = @inbounds model_fields.T[i, j, k]
    
        return -1 / timescale * mask(y, p) * (T - target_T)
    end

    FT = Forcing(temperature_relaxation, discrete_form = true, parameters = parameters)

    # closure

    κh = 0.5e-5 # [m²/s] horizontal diffusivity
    νh = 30.0   # [m²/s] horizontal viscocity
    κz = 0.5e-5 # [m²/s] vertical diffusivity
    νz = 3e-4   # [m²/s] vertical viscocity

    horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
    vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz)

    vertical_closure_CATKE = CATKEVerticalDiffusivity(minimum_tke=1e-7,
                                                    maximum_tracer_diffusivity=0.3,
                                                    maximum_viscosity=0.5)

    @info "Building a model..."

    model = HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = SplitExplicitFreeSurface(substeps=500),
        momentum_advection = WENO(),
        tracer_advection = WENO(),
        buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType),constant_salinity=35),
        coriolis = coriolis,
        closure = (horizontal_closure, vertical_closure, vertical_closure_CATKE),
        tracers = (:T, :e),
        boundary_conditions = (T = T_bcs, u = u_bcs, v = v_bcs),
        forcing = (; T = FT)
    )

    model.clock.last_Δt = Δt₀

    return model
end

#####
##### Forward simulation (not actually using the Simulation struct)
#####

using Oceananigans: AbstractModel, fields, prognostic_fields, location, TendencyCallsite
using Oceananigans.Advection: U_dot_∇v
using Oceananigans.Coriolis
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!,
                                tick!,
                                compute_pressure_correction!,
                                correct_velocities_and_cache_previous_tendencies!,
                                step_lagrangian_particles!,
                                ab2_step!,
                                QuasiAdamsBashforth2TimeStepper,
                                compute_flux_bc_tendencies!,
                                compute_tendencies!,
                                ab2_step_field!

using Oceananigans.Biogeochemistry: update_biogeochemical_state!, update_tendencies!
using Oceananigans.BoundaryConditions: update_boundary_conditions!, compute_x_bcs!, compute_y_bcs!, compute_z_bcs!, compute_z_bottom_bc!, compute_z_top_bc!, getbc
using Oceananigans.TurbulenceClosures: compute_diffusivities!, implicit_step!,
                                       is_vertically_implicit, ivd_diagonal, _ivd_lower_diagonal,
                                       _ivd_upper_diagonal, _implicit_linear_coefficient, ivd_diffusivity, getclosure,
                                       κzᶜᶜᶠ, κᶜᶜᶠ, diffusivity, ∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₂ⱼ

using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node, materialize_immersed_boundary,
                                       compute_numerical_bottom_height!, get_active_cells_map, immersed_cell, _immersed_cell

using Oceananigans.Models: update_model_field_time_series!, initialization_update_state!,
                           initialize_immersed_boundary_grid!, interior_tendency_kernel_parameters,
                           complete_communication_and_compute_buffer!

using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.Fields: tupled_fill_halo_regions!, immersed_boundary_condition, instantiated_location
using Oceananigans.Solvers: solve!, solve_batched_tridiagonal_system_kernel!, solve_batched_tridiagonal_system_z!, get_coefficient
using Oceananigans.Operators: σⁿ, σ⁻, Δz⁻¹, ℑzᵃᵃᶠ, ∂yᶜᶠᶜ, Az, volume, active_weighted_ℑxyᶜᶠᶜ, ℑxyᶜᶠᵃ, not_peripheral_node, ℑyᵃᶠᵃ, ℑxᶜᵃᵃ

using Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!, p_kernel_parameters
using Oceananigans.Models.HydrostaticFreeSurfaceModels: mask_immersed_model_fields!,
                                                        compute_w_from_continuity!,
                                                        w_kernel_parameters,
                                                        update_grid_vertical_velocity!,
                                                        _compute_w_from_continuity!,
                                                        compute_free_surface_tendency!,
                                                        scale_by_stretching_factor!,
                                                        ab2_step_grid!,
                                                        ab2_step_velocities!,
                                                        ab2_step_tracers!,
                                                        step_free_surface!,
                                                        _ab2_step_tracer_field!,
                                                        compute_hydrostatic_free_surface_tendency_contributions!,
                                                        compute_hydrostatic_momentum_tendencies!,
                                                        compute_hydrostatic_free_surface_Gc!,
                                                        compute_hydrostatic_free_surface_Gu!,
                                                        compute_hydrostatic_free_surface_Gv!,
                                                        explicit_barotropic_pressure_y_gradient,
                                                        grid_slope_contribution_y,
                                                        hydrostatic_fields,
                                                        compute_flux_bcs!

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: get_top_tracer_bcs, 
                                                                     update_previous_compute_time!,
                                                                     time_step_catke_equation!,
                                                                     compute_average_surface_buoyancy_flux!,
                                                                     compute_CATKE_diffusivities!,
                                                                     mask_diffusivity,
                                                                     κuᶜᶜᶠ, κcᶜᶜᶠ, κeᶜᶜᶠ,
                                                                     TKE_mixing_lengthᶜᶜᶠ, turbulent_velocityᶜᶜᶜ,
                                                                     convective_length_scaleᶜᶜᶠ, stability_functionᶜᶜᶠ,
                                                                     stable_length_scaleᶜᶜᶠ



using KernelAbstractions: @kernel, @index

function loop!(arch, grid, v, Gv, Δt)

    launch!(arch, grid, :xyz,
            _bad_compute_hydrostatic_free_surface_Gv!, Gv, grid; active_cells_map=nothing)

    launch!(arch, grid, :xyz,
            _bad_ab2_step_field!, v, Δt, Gv)
    return nothing
end

@kernel function _bad_ab2_step_field!(u, Δt, Gⁿ)
    i, j, k = @index(Global, NTuple)
    @inbounds u[i, j, k] += Δt * Gⁿ[i, j, k]
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function _bad_compute_hydrostatic_free_surface_Gv!(Gv, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = -1e6 * ridge_check(i, j, k, grid)
end

@inline function ridge_check(i, j, k, grid)
    # Doesn't work:
    
    active_nodes = (!(grid.underlying_grid.z.cᵃᵃᶜ[k] ≤ grid.immersed_boundary.bottom_height[i, j-1, 1]) # NEGATING THIS ELIMINATES ERROR, NEED TO INVESTIGATE
                  & !(grid.underlying_grid.z.cᵃᵃᶜ[k] ≤ grid.immersed_boundary.bottom_height[i-1, j-1, 1])
                  & !(grid.underlying_grid.z.cᵃᵃᶜ[k] ≤ grid.immersed_boundary.bottom_height[i+1, j-1, 1])
                  & (j > 1)
                  & (j < (grid.Ny+2))
                  & (k > 0)
                  & (k < grid.Nz+1))
    
    #=
    # Works:
    active_nodes = ((grid.underlying_grid.z.cᵃᵃᶜ[k] > grid.immersed_boundary.bottom_height[i, j-1, 1])
                  & (grid.underlying_grid.z.cᵃᵃᶜ[k] > grid.immersed_boundary.bottom_height[i-1, j-1, 1])
                  & (grid.underlying_grid.z.cᵃᵃᶜ[k] > grid.immersed_boundary.bottom_height[i+1, j-1, 1])
                  & (j > 1)
                  & (j < (grid.Ny+2))
                  & (k > 0)
                  & (k < grid.Nz+1))
    =#
    
    mask = active_nodes == 0
    return ifelse(mask, zero(grid), 100.0 / active_nodes)
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
arch = ReactantState()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
grid    = make_grid(arch, Nx, Ny, Nz, Δz_center)
vfield  = Field{Center, Face, Center}(grid)
Gvfield = Field{Center, Face, Center}(grid)

@info "Built model."

@info "Vanilla model as a comparison..."

# Architecture
varch = CPU()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
vgrid    = make_grid(varch, Nx, Ny, Nz, Δz_center)
vvfield  = Field{Center, Face, Center}(vgrid)
vGvfield = Field{Center, Face, Center}(vgrid)

using InteractiveUtils

@info "Comparing the pre-run model states..."

throw_error = false
include_halos = false
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

@allowscalar @show maximum(abs.(convert(Array, vfield) - convert(Array, vvfield)))
@allowscalar @show maximum(abs.(convert(Array, Gvfield) - convert(Array, vGvfield)))

@info "Running the vanilla model"
tic       = time()
loop!(varch, vgrid, vvfield, vGvfield, Δt₀)
vrun_toc  = time() - tic
@show vrun_toc

@info "Compiling the model run..."
tic = time()
rloop! = @compile raise_first=true raise=true sync=true loop!(arch, grid, vfield, Gvfield, Δt₀)
compile_toc = time() - tic

@show compile_toc

@info "Running the Reactant model"

tic      = time()
rloop!(arch, grid, vfield, Gvfield, Δt₀)
rrun_toc = time() - tic
@show rrun_toc

@info "Comparing the model states after running..."

@allowscalar @show maximum(abs.(convert(Array, vfield) - convert(Array, vvfield)))
@allowscalar @show maximum(abs.(convert(Array, Gvfield) - convert(Array, vGvfield)))