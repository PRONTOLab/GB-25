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

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
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

using InteractiveUtils

using KernelAbstractions: @kernel, @index

using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!, w_kernel_parameters, _compute_w_from_continuity!
using Oceananigans.Utils: launch!

using Oceananigans.Architectures: device
using Oceananigans.Grids: topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, div_xyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_cell, _immersed_cell

@show @which immersed_cell(10, 10, 9, grid.underlying_grid, grid.immersed_boundary)

bad_compute_w_from_continuity!(w, arch, grid; parameters = w_kernel_parameters(grid)) =
    launch!(arch, grid, parameters, _bad_compute_w_from_continuity!, w, grid)

@kernel function _bad_compute_w_from_continuity!(w, grid)
    i, j = @index(Global, NTuple)

    wᵏ = zero(eltype(w))
    @inbounds w[i, j, 1] = wᵏ

    Nz = size(grid, 3)
    for k in 2:Nz+1
        not_immersed = !_immersed_cell(i, j, k-1, grid.underlying_grid, grid.immersed_boundary)

        wᵏ -= 0.01 * not_immersed
        @inbounds w[i, j, k] = wᵏ
    end
end

w = Field{Center, Center, Face}(grid)

tic = time()
rcompute_w_from_continuity! = @compile raise_first=true raise=true sync=true bad_compute_w_from_continuity!(w, architecture, grid; parameters=w_kernel_parameters(grid))
compile_toc = time() - tic

@show compile_toc

@info "Running the simulation..."

tic = time()
rcompute_w_from_continuity!(w, architecture, grid; parameters=w_kernel_parameters(grid))
run_toc = time() - tic

@show run_toc