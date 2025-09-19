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
using Oceananigans.Grids: xnode, ynode, znode, XDirection, YDirection, ZDirection, peripheral_node, static_column_depthᶜᶜᵃ, rnode, getnode, inactive_cell
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity

using SeawaterPolynomials

#using Oceananigans.Architectures: GPU
#using CUDA
#CUDA.device!(0)

using Reactant
using GordonBell25
using Oceananigans.Architectures: ReactantState, architecture
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
##### Forward simulation (not actually using the Simulation struct)
#####

using KernelAbstractions: @kernel, @index

using Oceananigans.Utils: launch!

function loop!(arch, grid, Gv, z)

    launch!(arch, grid, :xyz,
            _bad_compute_hydrostatic_free_surface_Gv!, Gv, z; active_cells_map=nothing)

    return nothing
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function _bad_compute_hydrostatic_free_surface_Gv!(Gv, z)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = ridge_check(k, z)
end

@inline function ridge_check(k, z)
    # Doesn't work:
    active_nodes = !(z[k] ≤ -1500)
    # Works:
    #active_nodes = z[k] > -1500
    
    return active_nodes
end

#####
##### Actually creating our model and using these functions to run it:
#####

@info "Vanilla model as a comparison..."

# "Vanilla" data:
varch    = CPU()
vgrid    = make_grid(varch, Nx, Ny, Nz, Δz_center)
vGvfield = zeros((8, 9, 8))
vz       = [-185.99366913029345, -166.00501682133535, -146.0163645123772, -126.0277122034191, -106.03905989446096, -86.99190207831609, -69.73904173760515, -54.111450849280025,
            -39.95602432000002, -27.13408000000002, -15.520000000000017, -5.000000000000007, 5.000000000000007, 15.000000000000021, 25.000000000000036, 35.00000000000005]

# Architecture
arch    = ReactantState()
grid    = make_grid(arch, Nx, Ny, Nz, Δz_center)
Gvfield = Reactant.ConcreteRArray(vGvfield)
z       = Reactant.ConcreteRArray(vz)

@info "Built model."

using InteractiveUtils

@info "Comparing the pre-run model states..."

@allowscalar @show maximum(abs.(convert(Array, Gvfield) - convert(Array, vGvfield)))

@info "Running the vanilla model"
tic       = time()
loop!(varch, vgrid, vGvfield, vz)
vrun_toc  = time() - tic
@show vrun_toc

@info "Compiling the model run..."
tic = time()
thing = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-affine-to-stablehlo{prefer_while_raising=false},canonicalize,arith-raise{stablehlo=true}"
rloop! = @compile raise_first=true raise=thing sync=true loop!(arch, grid, Gvfield, z)
compile_toc = time() - tic

@show compile_toc

@info "Running the Reactant model"

tic      = time()
rloop!(arch, grid, Gvfield, z)
rrun_toc = time() - tic
@show rrun_toc

@info "Comparing the model states after running..."

@allowscalar @show maximum(abs.(convert(Array, Gvfield) - convert(Array, vGvfield)))