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

using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_tendency_contributions!


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

halo_size = 4 #3 for non-immersed grid


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

function build_model(grid)

    @info "Building a model..."

    @show @which HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = nothing,
        momentum_advection = nothing,
        tracer_advection = nothing,
        buoyancy = nothing,
        tracers = nothing
    )

    model = HydrostaticFreeSurfaceModel(
        grid = grid,
        free_surface = nothing,
        momentum_advection = nothing,
        tracer_advection = nothing,
        buoyancy = nothing,
        tracers = nothing
    )

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

mutable struct MyModel{G, T}
    grid::G  
    clock::Clock{T}
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture
architecture = ReactantState()

# Make the grid:
grid   = make_grid(architecture, Nx, Ny, Nz, z_faces)
model  = build_model(grid)
dmodel = Enzyme.make_zero(model)

mymodel = MyModel(model.grid, model.clock)
dmymodel = Enzyme.make_zero(mymodel)

@info "Compiling the model run..."
tic = time()
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true  differentiate_tracer_error(model, dmodel)
compile_toc = time() - tic

@show compile_toc
