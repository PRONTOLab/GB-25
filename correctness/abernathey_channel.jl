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
using Oceananigans.Grids: xnode, ynode, znode, XDirection, YDirection, ZDirection, peripheral_node
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
Nx = 16
Ny = 16
Nz = 16

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
        free_surface = SplitExplicitFreeSurface(substeps=50),
        momentum_advection = nothing,
        tracer_advection = WENO(),
        buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType),constant_salinity=35),
        coriolis = nothing,
        closure = (horizontal_closure, vertical_closure, vertical_closure_CATKE),
        tracers = (:T, :e)
    )

    model.clock.last_Δt = Δt₀

    return model
end

#####
##### Special initial and boundary conditions
#####

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
using Oceananigans.Models: update_model_field_time_series!
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

using InteractiveUtils

using KernelAbstractions: @kernel, @index

function run_reentrant_channel_model!(model, Tᵢ)
    # setting IC's and BC's:
    set!(model.tracers.T, Tᵢ)

    # Step it forward
    launch!(model.grid.architecture, model.grid, :xy,
            a_kernel!,
            model.tracers.T,
            size(model.grid, 3),
            model.diffusivity_fields[1]._tupled_tracer_diffusivities[1])

    return nothing
end

@kernel function a_kernel!(f, Nz, K)
    i, j = @index(Global, NTuple)
    
    for k = 2:Nz
        @inbounds f[i, j, k] = f[i, j, k] + K[i, j, k-1] * f[i, j, k-1]
    end
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
vTᵢ          = temperature_init(vmodel.grid, parameters)


N = length(model.closure)
vi_closure            = Tuple(model.closure[n]            for n = 1:N if is_vertically_implicit(model.closure[n]))
vi_diffusivity_fields = Tuple(model.diffusivity_fields[n] for n = 1:N if is_vertically_implicit(model.closure[n]))
@show @which getclosure(1, 1, vi_closure[1])

vN = length(vmodel.closure)
vvi_closure            = Tuple(vmodel.closure[n]            for n = 1:vN if is_vertically_implicit(vmodel.closure[n]))
vvi_diffusivity_fields = Tuple(vmodel.diffusivity_fields[n] for n = 1:vN if is_vertically_implicit(vmodel.closure[n]))
@show @which getclosure(1, 1, vvi_closure[1])

@show @which diffusivity(vi_closure[1], vi_diffusivity_fields[1], Val(1))
@show @which diffusivity(vvi_closure[1], vvi_diffusivity_fields[1], Val(1))

@show @which κᶜᶜᶠ(1, 1, 1, model.timestepper.implicit_solver.grid, (Center(), Center(), Center()), diffusivity(vi_closure[1], vi_diffusivity_fields[1], Val(1)), Val(1), model.clock)
@show @which κᶜᶜᶠ(1, 1, 1, vmodel.timestepper.implicit_solver.grid, (Center(), Center(), Center()), diffusivity(vvi_closure[1], vvi_diffusivity_fields[1], Val(1)), Val(1), vmodel.clock)

@show @which ℑzᵃᵃᶠ(1, 1, 1, model.timestepper.implicit_solver.grid, vi_diffusivity_fields[1]._tupled_tracer_diffusivities[1])
@show @which ℑzᵃᵃᶠ(1, 1, 1, vmodel.timestepper.implicit_solver.grid, vvi_diffusivity_fields[1]._tupled_tracer_diffusivities[1])

@info "Comparing the pre-run model states..."

throw_error = false
include_halos = false
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

GordonBell25.compare_states(model, vmodel; include_halos, throw_error, rtol, atol)

@info "Compiling the model run..."
tic = time()
rrun_reentrant_channel_model! = @compile raise_first=true raise=true sync=true run_reentrant_channel_model!(model, Tᵢ)
compile_toc = time() - tic

@show compile_toc

@info "Running the simulations..."

tic      = time()
rrun_reentrant_channel_model!(model, Tᵢ)
rrun_toc = time() - tic
@show rrun_toc

tic       = time()
run_reentrant_channel_model!(vmodel, vTᵢ)
vrun_toc  = time() - tic
@show vrun_toc

@info "Comparing the model states after running..."

GordonBell25.compare_states(model, vmodel; include_halos, throw_error, rtol, atol)
