using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using Oceananigans.Solvers.KernelAbstractions
using Reactant

using ClimaOcean
using OrthogonalSphericalShellGrids: TripolarGrid

@kernel function _compute_numerical_bottom_height!(_, _, _)
    nothing
end

function MyImmersedBoundaryGrid(arch, grid)
    Oceananigans.Utils.launch!(arch, grid, :xy, _compute_numerical_bottom_height!, nothing, nothing, nothing)
    return nothing
end

function _grid(arch::Architectures.AbstractArchitecture)
    resolution = 2
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 170 / resolution)
    Nz = 20
    Δt = 1minutes
    z_faces = exponential_z_faces(; Nz, depth=4000, h=30)
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    return MyImmersedBoundaryGrid(arch, underlying_grid)
end
