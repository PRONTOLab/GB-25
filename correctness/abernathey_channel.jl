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
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

#using Oceananigans.Architectures: GPU
#using CUDA
#CUDA.device!(0)

using Reactant
using GordonBell25
using Oceananigans.Architectures: ReactantState
Reactant.set_default_backend("cpu")

Oceananigans.defaults.FloatType = Float64

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]

# Architecture
architecture = ReactantState() #GPU()

# number of grid points
Nx = 96  # LowRes: 48
Ny = 192 # LowRes: 96
Nz = 32

halo_size = 4 #3 for non-immersed grid

# stretched grid
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)

const Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0

underlying_grid = RectilinearGrid(architecture,
    topology = (Periodic, Bounded, Bounded),
    size = (Nx, Ny, Nz),
    halo = (halo_size, halo_size, halo_size),
    x = (0, Lx),
    y = (0, Ly),
    z = (-Lz, 0))


# full ridge function:
function ridge_function(x, y)
    zonal = (Lz+100)exp(-(x - Lx/2)^2/(1e6kilometers))
    gap   = 1 - 0.5(tanh((y - (Ly/6))/1e5) - tanh((y - (Ly/2))/1e5))
    return zonal * gap - Lz
end

# Make into a ridge array:
ridge = Field{Center, Center, Nothing}(underlying_grid)
set!(ridge, ridge_function)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))

@info "Built a grid."

#@allowscalar @show grid

#####
##### Boundary conditions
#####

α = 2e-4     # [K⁻¹] thermal expansion coefficient
g = 9.8061   # [m/s²] gravitational constant
cᵖ = 3994.0   # [J/K]  heat capacity
ρ = 999.8    # [kg/m³] reference density

parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,            # buoyancy flux magnitude [m² s⁻³]
    y_shutoff = 5 / 6 * Ly,                # shutoff location for buoyancy flux [m]
    τ = 0.2 / ρ,                           # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
    H = Lz,                              # domain depth [m]
    h = 1000.0,                          # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,               # southern boundary of sponge layer [m]
    λt = 7.0days                         # relaxation time scale [s]
)

@inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
end

buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)


@inline function u_stress(i, j, grid, clock, model_fields, p)
    y = ynode(j, grid, Center())
    return -p.τ * sin(π * y / p.Ly)
end

u_stress_bc = FluxBoundaryCondition(u_stress, discrete_form = true, parameters = parameters)

@inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
@inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)

u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
v_bcs = FieldBoundaryConditions(bottom = v_drag_bc)

#####
##### Coriolis
#####

const f = -1e-4
const β = 1e-11
coriolis = BetaPlane(f₀ = f, β = β)

#####
##### Forcing and initial condition
#####

@inline initial_buoyancy(z, p) = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
@inline mask(y, p) = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)


@inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
    timescale = p.λt
    y = ynode(j, grid, Center())
    z = znode(k, grid, Center())
    target_b = initial_buoyancy(z, p)
    b = @inbounds model_fields.b[i, j, k]

    return -1 / timescale * mask(y, p) * (b - target_b)
end

Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)

# closure

κh = 0.5e-5 # [m²/s] horizontal diffusivity
νh = 30.0   # [m²/s] horizontal viscocity
κz = 0.5e-5 # [m²/s] vertical diffusivity
νz = 3e-4   # [m²/s] vertical viscocity

horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz)

#convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
#    convective_νz = 0.0)

vertical_closure_CATKE = CATKEVerticalDiffusivity(minimum_tke=1e-7,
                                                  maximum_tracer_diffusivity=0.3,
                                                  maximum_viscosity=0.5)


#####
##### Model building
#####

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = SplitExplicitFreeSurface(substeps=500),
    momentum_advection = WENO(),
    tracer_advection = WENO(),
    buoyancy = BuoyancyTracer(),
    coriolis = coriolis,
    closure = (horizontal_closure, vertical_closure, vertical_closure_CATKE),
    tracers = (:b, :e),
    boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs),
    forcing = (; b = Fb)
)

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
ε(σ) = σ * randn()
bᵢ(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h)) + ε(1e-8)

set!(model, b = bᵢ)

#####
##### Simulation building
#####
Δt₀ = 5minutes

@info "Compiling the simulation..."

model.clock.last_Δt = Δt₀

function loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = true track_numbers = false for i = 1:10
        time_step!(model, Δt)
    end
    return nothing
end

using InteractiveUtils

using KernelAbstractions: @kernel, @index

using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!, w_kernel_parameters, _compute_w_from_continuity!
using Oceananigans.Utils: launch!

using Oceananigans.Architectures: device
using Oceananigans.Grids: halo_size, topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, div_xyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_cell

bad_compute_w_from_continuity!(velocities, arch, grid; parameters = w_kernel_parameters(grid)) =
    launch!(arch, grid, parameters, _bad_compute_w_from_continuity!, velocities, grid)

@kernel function _bad_compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)

    u, v, w = U
    wᵏ = zero(eltype(w))
    @inbounds w[i, j, 1] = wᵏ

    Nz = size(grid, 3)
    for k in 2:Nz+1
        δ = flux_div_xyᶜᶜᶜ(i, j, k-1, grid, u, v) * Az⁻¹ᶜᶜᶜ(i, j, k-1, grid)

        # We do not account for grid changes in immersed cells
        not_immersed = !immersed_cell(i, j, k-1, grid)
        w̃ = Δrᶜᶜᶜ(i, j, k-1, grid) * ∂t_σ(i, j, k-1, grid) * not_immersed

        wᵏ -= (δ + w̃)
        @inbounds w[i, j, k] = wᵏ
    end
end

tic = time()
rcompute_w_from_continuity! = @compile raise_first=true raise=true sync=true compute_w_from_continuity!(model.velocities, model.architecture, model.grid; parameters=w_kernel_parameters(model.grid))
compile_toc = time() - tic

@show compile_toc

@info "Running the simulation..."

tic = time()
rcompute_w_from_continuity!(model.velocities, model.architecture, model.grid; parameters=w_kernel_parameters(model.grid))
run_toc = time() - tic

@show run_toc