ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8 --xla_dump_to=xla_dumps/super_simple_simulation_sharded_2/"
ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans
using Reactant

arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition=Partition(2, 2, 1)
)
Nx = Ny = Nz = 128
Nz = 16
grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=(0, 1));

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

gaussian_islands(λ, φ) = 2 * (mtn₁(λ, φ) + mtn₂(λ, φ))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_islands))

c = CenterField(grid)
∇²c = Field(∂x(∂x(c)) + ∂y(∂y(c)))

cᵢ(λ, φ, z) = exp(-(λ^2 + φ^2) / 200)
set!(c, cᵢ)
Oceananigans.BoundaryConditions.fill_halo_regions!(c)
compute!(∇²c)
parent(∇²c)
