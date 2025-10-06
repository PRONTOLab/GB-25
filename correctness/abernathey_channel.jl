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

using ClimaOcean.Diagnostics: MixedLayerDepthField

#using Oceananigans.Architectures: GPU
#using CUDA
#CUDA.device!(0)

using Reactant
using GordonBell25
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

using Enzyme

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
const Nx = 48 #96  # LowRes: 48
const Ny = 96 #192 # LowRes: 96
const Nz = 32

const x_midpoint = Int(Nx / 2) + 1

# stretched grid
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0

Δz = z_faces[2:end] - z_faces[1:end-1]

Δz = reshape(Δz, 1, :)

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
    zonal = (Lz+300)exp(-(x - Lx/2)^2/(1e6kilometers))
    gap   = 1 - 0.5(tanh((y - (Ly/6))/1e5) - tanh((y - (Ly/2))/1e5))
    return zonal * gap - Lz
end

function make_grid(architecture, Nx, Ny, Nz, z_faces)

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
    v_stress_bc = FluxBoundaryCondition(Field{Center, Face, Nothing}(grid))

    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    T_bcs = FieldBoundaryConditions(top = temperature_flux_bc)

    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(top = v_stress_bc, bottom = v_drag_bc)

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
        free_surface = SplitExplicitFreeSurface(substeps=10),
        momentum_advection = WENO(),
        tracer_advection = WENO(),
        buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType),constant_salinity=35),
        coriolis = coriolis,
        closure = (horizontal_closure, vertical_closure), #, vertical_closure_CATKE),
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
function u_wind_stress_init(grid, p)
    @inline u_stress(x, y) = -p.τ * sin(π * y / p.Ly)
    wind_stress = Field{Face, Center, Nothing}(grid)
    set!(wind_stress, u_stress)
    return wind_stress
end

function v_wind_stress_init(grid, p)
    wind_stress = Field{Center, Face, Nothing}(grid)
    set!(wind_stress, 0)
    return wind_stress
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
    @trace mincut = true checkpointing = true track_numbers = false for i = 1:169
        time_step!(model, Δt)
    end
    return nothing
end

function run_reentrant_channel_model!(model, Tᵢ, u_wind_stress, v_wind_stress)
    # setting IC's and BC's:
    set!(model.velocities.u.boundary_conditions.top.condition, u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, v_wind_stress)
    set!(model.tracers.T, Tᵢ)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0

    # Step it forward
    loop!(model)

    return nothing
end

function estimate_tracer_error(model, initial_temperature, u_wind_stress, v_wind_stress, Δz, mld)
    run_reentrant_channel_model!(model, initial_temperature, u_wind_stress, v_wind_stress)
    
    Nx, Ny, Nz = size(model.grid)

    # Compute the mean mixed layer depth:
    #compute!(mld)
    #avg_mld = sum(parent(mld)) / (Nx * Ny)

    # Alternatively, compute the zonal transport:
    zonal_transport = (model.velocities.u[x_midpoint,1:Ny,1:Nz] .* model.grid.Δyᵃᶜᵃ) .* Δz

    return sum(zonal_transport) / 1e6 # Put it in Sverdrups
end

function differentiate_tracer_error(model, Tᵢ, u_wind_stress, v_wind_stress, Δz, mld,
                                   dmodel, dTᵢ, du_wind_stress, dv_wind_stress, dΔz, dmld)

    dedν = autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(u_wind_stress, du_wind_stress),
                    Duplicated(v_wind_stress, dv_wind_stress),
                    Duplicated(Δz, dΔz),
                    Duplicated(mld, dmld))

    return dedν, du_wind_stress, dTᵢ
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState() #GPU()

# Timestep size:
Δt₀ = 5minutes 

# Make the grid:
grid          = make_grid(architecture, Nx, Ny, Nz, z_faces)
model         = build_model(grid, Δt₀, parameters)
u_wind_stress = u_wind_stress_init(model.grid, parameters)
v_wind_stress = v_wind_stress_init(model.grid, parameters)
Tᵢ            = temperature_init(model.grid, parameters)
mld           = MixedLayerDepthField(model.buoyancy, model.grid, model.tracers)
Δz            = Reactant.ConcreteRArray(Δz)

@info "Built $model."

dmodel         = Enzyme.make_zero(model)
dTᵢ            = Field{Center, Center, Center}(model.grid)
du_wind_stress = Field{Face, Center, Nothing}(model.grid)
dv_wind_stress = Field{Center, Face, Nothing}(model.grid)
dmld           = MixedLayerDepthField(dmodel.buoyancy, dmodel.grid, dmodel.tracers)
dΔz            = Enzyme.make_zero(Δz)

# Trying zonal transport:
#@allowscalar zonal_transport = Field(Integral(model.velocities.u, dims=(2,3)))

@info "Compiling the model run..."
tic = time()
#restimate_tracer_error = @compile raise_first=true raise=true compile_options=CompileOptions(; disable_auto_batching_passes=true) sync=true estimate_tracer_error(model, Tᵢ, u_wind_stress, v_wind_stress, Δz, mld)
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true  differentiate_tracer_error(model, Tᵢ, u_wind_stress, v_wind_stress, Δz, mld, dmodel, dTᵢ, du_wind_stress, dv_wind_stress, dΔz, dmld)
compile_toc = time() - tic

@show compile_toc

@info "Running the simulation..."

using FileIO, JLD2

graph_directory = "run_abernathy_model_ad_169steps_noCATKE/"
filename        = graph_directory * "data_init.jld2"

if !isdir(graph_directory) Base.Filesystem.mkdir(graph_directory) end

if isa(model.grid, ImmersedBoundaryGrid)
    bottom_height = model.grid.immersed_boundary.bottom_height
else
    bottom_height = Field{Center, Center, Nothing}(model.grid)
    set!(bottom_height, -Lz)
end

jldsave(filename; Nx, Ny, Nz,
                  bottom_height=convert(Array, interior(bottom_height)),
                  T_init=convert(Array, interior(model.tracers.T)),
                  e_init=convert(Array, interior(model.tracers.e)),
                  u_wind_stress=convert(Array, interior(u_wind_stress)),
                  v_wind_stress=convert(Array, interior(v_wind_stress)))

tic = time()
#output = restimate_tracer_error(model, Tᵢ, wind_stress, Δz, mld)
dedν, du_wind_stress, dTᵢ = rdifferentiate_tracer_error(model, Tᵢ, u_wind_stress, v_wind_stress, Δz, mld, dmodel, dTᵢ, du_wind_stress, dv_wind_stress, dΔz, dmld)
run_toc = time() - tic

@show run_toc
#@show output

@show dedν

filename = graph_directory * "data_final.jld2"

jldsave(filename; Nx, Ny, Nz,
                  T_final=convert(Array, interior(model.tracers.T)),
                  e_final=convert(Array, interior(model.tracers.e)),
                  ssh=convert(Array, interior(model.free_surface.η)),
                  u=convert(Array, interior(model.velocities.u)),
                  v=convert(Array, interior(model.velocities.v)),
                  w=convert(Array, interior(model.velocities.w)),
                  mld=convert(Array, interior(mld)),
                  #zonal_transport=convert(Float64, output))
                  zonal_transport=convert(Float64, dedν[2]),
                  du_wind_stress=convert(Array, interior(du_wind_stress)),
                  dv_wind_stress=convert(Array, interior(dv_wind_stress)),
                  dT=convert(Array, interior(dTᵢ)))

