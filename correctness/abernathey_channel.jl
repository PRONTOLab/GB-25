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

using Reactant
using GordonBell25
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

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
    y_shutoff = 5 / 6 * Ly,                # shutoff location for buoyancy flux [m]
    τ = 0.2 / ρ,                           # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
    H = Lz,                              # domain depth [m]
    h = 1000.0,                          # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,               # southern boundary of sponge layer [m]
    λt = 7.0days                         # relaxation time scale [s]
)

function make_grid(architecture, Nx, Ny, Nz, Δz_center)
    z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
    z_faces[Nz+1] = 0

    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = (-Lz, 0))

    return underlying_grid
end

#####
##### Model construction:
#####

function build_model(grid, Δt₀, parameters)

    @inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
        y = ynode(j, grid, Center())
        return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
    end

    buoyancy_flux_bc = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)

    u_stress_bc = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

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
    coriolis = BetaPlane(f₀ = f, β = β)

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
        buoyancy = BuoyancyTracer(),
        coriolis = coriolis,
        closure = (horizontal_closure, vertical_closure, vertical_closure_CATKE),
        tracers = (:b, :e),
        boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs)
    )

    model.clock.last_Δt = Δt₀

    return model
end

#####
##### Forward simulation (not actually using the Simulation struct)
#####

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: update_state!

function loop!(model)
    Δt = model.clock.last_Δt
    initialize!(model)
    update_state!(model)
    time_step!(model, Δt, euler=true)
    return nothing
end


#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState() #GPU()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
grid  = make_grid(architecture, Nx, Ny, Nz, Δz_center)
model = build_model(grid, Δt₀, parameters)

using InteractiveUtils

@show @which Oceananigans.TimeSteppers.first_time_step!(model, Δt₀)

@info "Compiling the model run..."
tic = time()
rloop! = @compile raise_first=true raise=true sync=true loop!(model)
compile_toc = time() - tic

@show compile_toc

rloop!(model)