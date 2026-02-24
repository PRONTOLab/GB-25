#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using InteractiveUtils

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, HorizontalFormulation

using SeawaterPolynomials

using Reactant
using CUDA
using Oceananigans.Architectures: ReactantState
#Reactant.set_default_backend("cpu")

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

using Enzyme

using Oceananigans.TimeSteppers: update_state!, compute_tendencies!
using Oceananigans.ImmersedBoundaries: get_active_cells_map
using Oceananigans.Utils: work_layout, KernelParameters
using Oceananigans.Biogeochemistry: update_tendencies!
using Oceananigans.Models: interior_tendency_kernel_parameters, complete_communication_and_compute_buffer!

using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_tendency_contributions!,
                                                        hydrostatic_velocity_fields

using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions


const Ntimesteps = 400

Oceananigans.defaults.FloatType = Float64

#
# Model parameters to set first:
#

# number of grid points
const Nx = 80  # LowRes: 48
const Ny = 160 # LowRes: 96
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

const halo_size = 4 #3 for non-immersed grid


function make_grid(architecture, Nx, Ny, Nz, z_faces)

    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    return underlying_grid
end

#####
##### Model construction:
#####

function my_hydrostatic_velocity_fields()

    total_size = (Nx+2*halo_size) * (Ny+2*halo_size) * (Nz+2*halo_size)

    u = Float64.(reshape(1:total_size, Nx+2*halo_size, Ny+2*halo_size, Nz+2*halo_size))
    v = Float64.(reshape(1:total_size, Nx+2*halo_size, Ny+2*halo_size, Nz+2*halo_size))
    w = Float64.(reshape(1:total_size, Nx+2*halo_size, Ny+2*halo_size, Nz+2*halo_size))

    u = Reactant.to_rarray(u)
    v = Reactant.to_rarray(v)
    w = Reactant.to_rarray(w)

    return (u=u, v=v, w=w)
end

function Clock_helper(; time,
               last_Δt = Inf,
               last_stage_Δt = Inf,
               iteration = 0,
               stage = 1)

    TT = typeof(time)
    DT = typeof(last_Δt)
    IT = typeof(iteration)
    last_stage_Δt = convert(DT, last_Δt)
    return Clock{TT, DT, IT, typeof(stage)}(time, last_Δt, last_stage_Δt, iteration, stage)
end

function my_Clock()
    FT = Oceananigans.defaults.FloatType
    t = ConcreteRNumber(zero(FT))
    iter = ConcreteRNumber(0)
    stage = 0
    last_Δt = convert(FT, Inf)
    last_stage_Δt = convert(FT, Inf)
    return Clock_helper(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

function build_model()

    clock = my_Clock()

    velocities = my_hydrostatic_velocity_fields()
    
    model = MyModel(clock, velocities)

    return model
end

#####
##### Forward simulation (not actually using the Simulation struct)
#####


function estimate_tracer_error(model)
    model.clock.iteration = 0

    return nothing
end

function differentiate_tracer_error(model, dmodel)

    dedν = autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel))

    return dedν
end

mutable struct MyModel{T, U}
    clock::Clock{T}
    velocities::U
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState()

# Make the grid:
model  = build_model()
dmodel = Enzyme.make_zero(model)

@info "Compiling the model run..."
tic = time()
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true  differentiate_tracer_error(model, dmodel)
compile_toc = time() - tic

@show compile_toc
