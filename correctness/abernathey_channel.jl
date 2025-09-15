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
using Oceananigans.Grids: xnode, ynode, znode, XDirection, YDirection, ZDirection, peripheral_node, static_column_depthᶜᶜᵃ, rnode, getnode
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

using SeawaterPolynomials

#using Oceananigans.Architectures: GPU
#using CUDA
#CUDA.device!(0)

using Reactant
using GordonBell25
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

using Enzyme


Oceananigans.defaults.FloatType = Float64

#
# Model parameters to set first:
#

# number of grid points
Nx = 48 #96  # LowRes: 48
Ny = 96 #192 # LowRes: 96
Nz = 32

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
##### Special initial and boundary conditions
#####

# wind stress:
function wind_stress_init(grid, p)
    @inline u_stress(x, y) = -p.τ * sin(π * y / p.Ly)
    wind_stress = Field{Face, Center, Nothing}(grid)
    set!(wind_stress, u_stress)
    return wind_stress
end

# resting initial condition
function temperature_init(grid, parameters)
    Tᵢ_function(x, y, z) = parameters.ΔT * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h))
    Tᵢ = Field{Center, Center, Center}(grid)
    set!(Tᵢ, Tᵢ_function)
    return Tᵢ
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
                                compute_tendencies!

using Oceananigans.Biogeochemistry: update_biogeochemical_state!, update_tendencies!
using Oceananigans.BoundaryConditions: update_boundary_conditions!
using Oceananigans.TurbulenceClosures: compute_diffusivities!, implicit_step!,
                                       is_vertically_implicit, ivd_diagonal, _ivd_lower_diagonal,
                                       _ivd_upper_diagonal, _implicit_linear_coefficient, ivd_diffusivity, getclosure,
                                       κzᶜᶜᶠ, κᶜᶜᶠ, diffusivity, ∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₂ⱼ

using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node, materialize_immersed_boundary, compute_numerical_bottom_height!, get_active_cells_map
using Oceananigans.Models: update_model_field_time_series!, initialization_update_state!,
                           initialize_immersed_boundary_grid!, interior_tendency_kernel_parameters,
                           complete_communication_and_compute_buffer!

using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.Fields: tupled_fill_halo_regions!, immersed_boundary_condition
using Oceananigans.Solvers: solve!, solve_batched_tridiagonal_system_kernel!, solve_batched_tridiagonal_system_z!, get_coefficient
using Oceananigans.Operators: σⁿ, σ⁻, Δz⁻¹, ℑzᵃᵃᶠ, ∂yᶜᶠᶜ

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
                                                        hydrostatic_fields

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

function loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = true track_numbers = false for i = 1:2
        bad_time_step!(model, Δt)
    end
    return nothing
end

function run_reentrant_channel_model!(model, Tᵢ, wind_stress)
    # setting IC's and BC's:
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)
    set!(model.tracers.T, Tᵢ)

    # Initialize the model
    model.clock.iteration = 1
    model.clock.time = 0

    # Step it forward
    loop!(model)

    return nothing
end

function bad_time_step!(model, Δt;
                    callbacks=[], euler=false)

    # Full step for tracers, fractional step for velocities.
    compute_flux_bc_tendencies!(model)
    ab2_step!(model, Δt)

    bad_update_state!(model, model.grid, callbacks; compute_tendencies=true)

    return nothing
end

function bad_update_state!(model, grid, callbacks; compute_tendencies = true)

    bad_compute_auxiliaries!(model)

    bad_compute_hydrostatic_momentum_tendencies!(model, model.velocities, :xyz; active_cells_map=nothing)

    return nothing
end

function bad_compute_auxiliaries!(model; w_parameters = w_kernel_parameters(model.grid),
                                         p_parameters = p_kernel_parameters(model.grid),
                                         κ_parameters = :xyz)

    grid        = model.grid
    closure     = model.closure
    tracers     = model.tracers
    diffusivity = model.diffusivity_fields
    buoyancy    = model.buoyancy

    P    = model.pressure.pHY′
    arch = grid.architecture

    # Update the vertical velocity to comply with the barotropic correction step
    update_grid_vertical_velocity!(model, grid, model.vertical_coordinate)

    # Advance diagnostic quantities
    compute_w_from_continuity!(model; parameters = w_parameters)
    update_hydrostatic_pressure!(P, arch, grid, buoyancy, tracers; parameters = p_parameters)

    # Update closure diffusivities
    compute_diffusivities!(diffusivity, closure, model; parameters = κ_parameters)

    return nothing
end

function bad_compute_hydrostatic_momentum_tendencies!(model, velocities, kernel_parameters; active_cells_map=nothing)

    grid = model.grid
    arch = grid.architecture

    v_immersed_bc = immersed_boundary_condition(velocities.v)

    v_forcing = model.forcing.v

    start_momentum_kernel_args = (model.advection.momentum,
                                  model.coriolis,
                                  model.closure)

    end_momentum_kernel_args = (velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.diffusivity_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.vertical_coordinate,
                                model.clock)

    v_kernel_args = tuple(start_momentum_kernel_args..., v_immersed_bc, end_momentum_kernel_args..., v_forcing)

    launch!(arch, grid, kernel_parameters,
            bad_compute_hydrostatic_free_surface_Gv!, model.timestepper.Gⁿ.v, grid, 
            v_kernel_args; active_cells_map)

    return nothing
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function bad_compute_hydrostatic_free_surface_Gv!(Gv, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = bad_hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid, args...)
end

@inline function bad_hydrostatic_free_surface_v_velocity_tendency(i, j, k, grid,
                                                              advection,
                                                              coriolis,
                                                              closure,
                                                              v_immersed_bc,
                                                              velocities,
                                                              free_surface,
                                                              tracers,
                                                              buoyancy,
                                                              diffusivities,
                                                              hydrostatic_pressure_anomaly,
                                                              auxiliary_fields,
                                                              ztype,
                                                              clock,
                                                              forcing)

    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers), auxiliary_fields)

    return ( - U_dot_∇v(i, j, k, grid, advection, velocities)
             - explicit_barotropic_pressure_y_gradient(i, j, k, grid, free_surface)
             - y_f_cross_U(i, j, k, grid, coriolis, velocities)
             - ∂yᶜᶠᶜ(i, j, k, grid, hydrostatic_pressure_anomaly)
             - grid_slope_contribution_y(i, j, k, grid, buoyancy, ztype, model_fields)
             - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, buoyancy)
             - immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid, velocities, v_immersed_bc, closure, diffusivities, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
grid        = make_grid(architecture, Nx, Ny, Nz, Δz_center)
model       = build_model(grid, Δt₀, parameters)
wind_stress = wind_stress_init(model.grid, parameters)
Tᵢ          = temperature_init(model.grid, parameters)


@info "Built $model."

@info "Vanilla model as a comparison..."

# Architecture
varchitecture = CPU()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
vgrid        = make_grid(varchitecture, Nx, Ny, Nz, Δz_center)
vmodel       = build_model(vgrid, Δt₀, parameters)
vwind_stress = wind_stress_init(vmodel.grid, parameters)
vTᵢ          = temperature_init(vmodel.grid, parameters)

using InteractiveUtils

@show @which compute_auxiliaries!(model)
@show @which compute_auxiliaries!(vmodel)

@show @which interior_tendency_kernel_parameters(model.grid.architecture, model.grid)
@show @which interior_tendency_kernel_parameters(vmodel.grid.architecture, vmodel.grid)

@info "Comparing the pre-run model states..."

throw_error = false
include_halos = false
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

GordonBell25.compare_states(model, vmodel; include_halos, throw_error, rtol, atol)

@info "Running the vanilla model"
tic       = time()
run_reentrant_channel_model!(vmodel, vTᵢ, vwind_stress)
vrun_toc  = time() - tic
@show vrun_toc

@info "Compiling the model run..."
tic = time()
rrun_reentrant_channel_model! = @compile raise_first=true raise=true sync=true run_reentrant_channel_model!(model, Tᵢ, wind_stress)
compile_toc = time() - tic

@show compile_toc

@info "Running the Reactant model"

tic      = time()
rrun_reentrant_channel_model!(model, Tᵢ, wind_stress)
rrun_toc = time() - tic
@show rrun_toc

@info "Comparing the model states after running..."

GordonBell25.compare_states(model, vmodel; include_halos, throw_error, rtol, atol)