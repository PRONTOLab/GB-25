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

using SeawaterPolynomials

using Reactant
using GordonBell25
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

#=
# https://github.com/CliMA/Oceananigans.jl/blob/c29939097a8d2f42966e930f2f2605803bf5d44c/src/AbstractOperations/binary_operations.jl#L5
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{Oceananigans.AbstractOperations.BinaryOperation{LX, LY, LZ, O, A, B, IA, IB, G, T}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, O, A, B, IA, IB, G, T}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)

    O2 = Reactant.traced_type_inner(O, seen, mode, track_numbers, sharding, runtime)

    A2 = Reactant.traced_type_inner(A, seen, mode, track_numbers, sharding, runtime)
    B2 = Reactant.traced_type_inner(B, seen, mode, track_numbers, sharding, runtime)
    IA2 = Reactant.traced_type_inner(IA, seen, mode, track_numbers, sharding, runtime)
    IB2 = Reactant.traced_type_inner(IB, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)

    T2 = eltype(G2)
    return Oceananigans.AbstractOperations.BinaryOperation{LX2, LY2, LZ2, O2, A2, B2, IA2, IB2, G2, T2}
end
=#
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
        z = (-Lz, 0))

    # Make into a ridge array:
    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, ridge_function)

    grid = underlying_grid #ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))
    return grid
end

#####
##### Model construction:
#####

function build_model(grid, Δt₀, parameters)

    @inline function buoyancy_flux(i, j, grid, clock, model_fields, p)
        y = ynode(j, grid, Center())
        return ifelse(y < p.y_shutoff, p.Qᵇ * cos(3π * y / p.Ly), 0.0)
    end

    @inline function temperature_flux(i, j, grid, clock, model_fields, p)
        y = ynode(j, grid, Center())
        return ifelse(y < p.y_shutoff, p.Qᵀ * cos(3π * y / p.Ly), 0.0)
    end

    buoyancy_flux_bc    = FluxBoundaryCondition(buoyancy_flux, discrete_form = true, parameters = parameters)
    temperature_flux_bc = FluxBoundaryCondition(temperature_flux, discrete_form = true, parameters = parameters)

    u_stress_bc = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    b_bcs = FieldBoundaryConditions(top = buoyancy_flux_bc)
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

    @inline initial_buoyancy(z, p)    = p.ΔB * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
    @inline initial_temperature(z, p) = p.ΔT * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
    @inline mask(y, p)                = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)


    @inline function buoyancy_relaxation(i, j, k, grid, clock, model_fields, p)
        timescale = p.λt
        y = ynode(j, grid, Center())
        z = znode(k, grid, Center())
        target_b = initial_buoyancy(z, p)
        b = @inbounds model_fields.b[i, j, k]

        return -1 / timescale * mask(y, p) * (b - target_b)
    end

    @inline function temperature_relaxation(i, j, k, grid, clock, model_fields, p)
        timescale = p.λt
        y = ynode(j, grid, Center())
        z = znode(k, grid, Center())
        target_T = initial_temperature(z, p)
        T = @inbounds model_fields.T[i, j, k]
    
        return -1 / timescale * mask(y, p) * (T - target_T)
    end

    Fb = Forcing(buoyancy_relaxation, discrete_form = true, parameters = parameters)
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
function buoyancy_init(grid, parameters)
    ε(σ) = σ * randn()
    bᵢ_function(x, y, z) = parameters.ΔB * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h)) + ε(1e-8)
    bᵢ = Field{Center, Center, Center}(grid)
    set!(bᵢ, bᵢ_function)
    return bᵢ
end

# resting initial condition
function temperature_init(grid, parameters)
    ε(σ) = σ * randn()
    Tᵢ_function(x, y, z) = parameters.ΔT * (exp(z / parameters.h) - exp(-Lz / parameters.h)) / (1 - exp(-Lz / parameters.h)) + ε(1e-8)
    Tᵢ = Field{Center, Center, Center}(grid)
    set!(Tᵢ, Tᵢ_function)
    return Tᵢ
end

#####
##### Forward simulation (not actually using the Simulation struct)
#####

function loop!(model)
    Δt = model.clock.last_Δt
    @trace mincut = true track_numbers = false for i = 1:10
        time_step!(model, Δt)
    end
    return nothing
end

function run_reentrant_channel_model!(model, Tᵢ, wind_stress)
    # setting IC's and BC's:
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)
    set!(model.tracers.T, Tᵢ)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0

    # Step it forward
    loop!(model)

    return nothing
end

function estimate_tracer_error(model, initial_temperature, wind_stress)
    run_reentrant_channel_model!(model, initial_temperature, wind_stress)
    # Compute the mean mixed layer depth:
    #compute!(mld)
    Nx, Ny, Nz = size(model.grid)
    #=
    avg_mld = 0.0
    
    for j0 = 1:Nx, i0 = 1:Ny
        @allowscalar avg_mld += @inbounds model.velocities.u[i0, j0, 1]^2
    end
    avg_mld = avg_mld / (Nx * Ny)
    =#
    # Hard way
    c² = parent(model.tracers.T).^2
    avg_mld = sum(c²) / (Nx * Ny * Nz)
    return avg_mld
end


#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState() #GPU()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
grid        = make_grid(architecture, Nx, Ny, Nz, Δz_center)
model       = build_model(grid, Δt₀, parameters)
wind_stress = wind_stress_init(model.grid, parameters)
Tᵢ          = temperature_init(model.grid, parameters)


@info "Built $model."

@info "Compiling the model run..."
tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(model, Tᵢ, wind_stress)
compile_toc = time() - tic

@show compile_toc


@info "Vanilla model as a comparison..."

# Architecture
varchitecture = CPU()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
vgrid        = make_grid(varchitecture, Nx, Ny, Nz, Δz_center)
vmodel       = build_model(vgrid, Δt₀, parameters)
vwind_stress = wind_stress_init(vmodel.grid, parameters)
vTᵢ          = buoyancy_init(vmodel.grid, parameters)

@info "Comparing the pre-run model states..."

throw_error = false
include_halos = false
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

GordonBell25.compare_states(model, vmodel; include_halos, throw_error, rtol, atol)

@info "Running the simulations..."

tic      = time()
avg_temp = restimate_tracer_error(model, Tᵢ, wind_stress)
rrun_toc = time() - tic
@show rrun_toc

tic       = time()
vavg_temp = estimate_tracer_error(vmodel, vTᵢ, vwind_stress)
vrun_toc  = time() - tic
@show vrun_toc

@info "Comparing the model states after running..."

GordonBell25.compare_states(model, vmodel; include_halos, throw_error, rtol, atol)