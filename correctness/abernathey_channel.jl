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

using InteractiveUtils

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

    # closure
    vertical_closure_CATKE = CATKEVerticalDiffusivity()

    @info "Building a model..."

    @show @which model = HydrostaticFreeSurfaceModel(
                        grid = grid,
                        free_surface = SplitExplicitFreeSurface(substeps=50),
                        momentum_advection = nothing,
                        tracer_advection = nothing,
                        buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(Oceananigans.defaults.FloatType),constant_salinity=35),
                        coriolis = nothing,
                        closure = vertical_closure_CATKE,
                        tracers = (:T, :e)
    )

    model = HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = SplitExplicitFreeSurface(substeps=50),
        momentum_advection = nothing,
        tracer_advection = nothing,
        buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(Oceananigans.defaults.FloatType),constant_salinity=35),
        coriolis = nothing,
        closure = vertical_closure_CATKE,
        tracers = (:T, :e)
    )

    return model
end

#####
##### Forward simulation (not actually using the Simulation struct)
#####

using Oceananigans: AbstractModel, fields, prognostic_fields, location
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

using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_conditions!
using Oceananigans.TurbulenceClosures: compute_diffusivities!, implicit_step!,
                                       is_vertically_implicit, ivd_diagonal, _ivd_lower_diagonal,
                                       _ivd_upper_diagonal, _implicit_linear_coefficient, ivd_diffusivity, getclosure,
                                       κzᶜᶜᶠ, κᶜᶜᶠ, diffusivity

using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models: update_model_field_time_series!, initialization_update_state!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.Fields: tupled_fill_halo_regions!
using Oceananigans.Solvers: solve!, solve_batched_tridiagonal_system_kernel!, solve_batched_tridiagonal_system_z!, get_coefficient
using Oceananigans.Operators: σⁿ, σ⁻, Δz⁻¹, ℑzᵃᵃᶠ

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
                                                        _ab2_step_tracer_field!

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


function bad_compute_diffusivities!(diffusivities, closure, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    
    launch!(arch, grid, parameters,
            bad_compute_CATKE_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)
    
    return nothing
end

@kernel function bad_compute_CATKE_diffusivities!(diffusivities, grid, closure, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    Jᵇ = diffusivities.Jᵇ

    # Note: we also compute the TKE diffusivity here for diagnostic purposes, even though it
    # is recomputed in time_step_turbulent_kinetic_energy.
    κu★ = κuᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ)
    κc★ = κcᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ)
    @inbounds κe★ = getnode(grid.underlying_grid.z.cᵃᵃᶠ, grid.Nz+1) - grid.immersed_boundary.bottom_height[i, j, 1]

    @inbounds begin
        diffusivities.κu[i, j, k] = κu★
        diffusivities.κc[i, j, k] = κc★
        diffusivities.κe[i, j, k] = κe★
    end
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState()
Δt₀ = 5minutes

# Make the grid:
grid        = make_grid(architecture, Nx, Ny, Nz, Δz_center)
model       = build_model(grid, Δt₀, parameters)
#diffusivity_fields = build_model(grid, Δt₀, parameters)

@info "Vanilla model as a comparison..."

# Architecture
varchitecture = CPU()

# Make the grid:
vgrid        = make_grid(varchitecture, Nx, Ny, Nz, Δz_center)
vmodel       = build_model(vgrid, Δt₀, parameters)
#vdiffusivity_fields = build_model(grid, Δt₀, parameters)

@info "Comparing the bottom heights:"
@allowscalar @show convert(Array, grid.immersed_boundary.bottom_height)
@show convert(Array, vgrid.immersed_boundary.bottom_height)

@allowscalar @show getnode(grid.underlying_grid.z.cᵃᵃᶠ, grid.Nz+1) - getnode(vgrid.underlying_grid.z.cᵃᵃᶠ, vgrid.Nz+1)

@info "Comparing the pre-init model states..."
@allowscalar @show maximum(abs.(convert(Array, model.diffusivity_fields.κe) - vmodel.diffusivity_fields.κe))
@allowscalar @show maximum(abs.(convert(Array, model.diffusivity_fields.κc) - vmodel.diffusivity_fields.κc))
@allowscalar @show maximum(abs.(convert(Array, model.diffusivity_fields.κu) - vmodel.diffusivity_fields.κu))


@show @which compute_diffusivities!(model.diffusivity_fields, model.closure, model; parameters = :xyz)
@show @which compute_diffusivities!(vmodel.diffusivity_fields, vmodel.closure, vmodel; parameters = :xyz)

@show @which rnode(1, 1, model.grid.Nz+1, model.grid.underlying_grid, Center(), Center(), Face())
@show @which rnode(1, 1, vmodel.grid.Nz+1, vmodel.grid.underlying_grid, Center(), Center(), Face())

# MLIR Error:
#rupdate_state! = @compile raise_first=true raise=true sync=true update_state!(model)

# What about this?
#rbad_compute_diffusivities! = @compile raise_first=true raise=true sync=true bad_compute_diffusivities!(model.diffusivity_fields, model.closure, model; parameters = :xyz)

bad_compute_diffusivities!(model.diffusivity_fields, model.closure, model; parameters = :xyz)
bad_compute_diffusivities!(vmodel.diffusivity_fields, vmodel.closure, vmodel; parameters = :xyz)

@info "And now comparing them again"
@allowscalar @show maximum(abs.(convert(Array, model.diffusivity_fields.κe) - vmodel.diffusivity_fields.κe))
@allowscalar @show maximum(abs.(convert(Array, model.diffusivity_fields.κc) - vmodel.diffusivity_fields.κc))
@allowscalar @show maximum(abs.(convert(Array, model.diffusivity_fields.κu) - vmodel.diffusivity_fields.κu))