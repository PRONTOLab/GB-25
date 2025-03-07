using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using Reactant

using CUDA
using KernelAbstractions

using ClimaOcean: exponential_z_faces
using OrthogonalSphericalShellGrids: TripolarGrid

function mtn₁(λ, φ)
    λ₁ = 70
    φ₁ = 55
    dφ = 5
    return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
end

function mtn₂(λ, φ)
    λ₁ = 70
    λ₂ = λ₁ + 180
    φ₂ = 55
    dφ = 5
    return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
end

function _grid(arch::Architectures.AbstractArchitecture)
    # Horizontal resolution
    resolution = 2 # 1/4 for quarter degree
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 170 / resolution)

    # Vertical resolution
    Nz = 20 # eventually we want to increase this to between 100-600

    # Time step. This must be decreased as resolution is decreased.
    Δt = 1minutes

    # Grid setup
    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz
    grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    zb = z_faces[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))
    ib = GridFittedBottom(gaussian_islands)

    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    Oceananigans.Utils.launch!(Oceananigans.Architectures.architecture(grid), grid, :xy, Oceananigans.ImmersedBoundaries._compute_numerical_bottom_height!, bottom_field, grid, ib)
end
