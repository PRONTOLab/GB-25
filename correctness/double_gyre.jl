ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using KernelAbstractions
using GordonBell25
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using InteractiveUtils
using KernelAbstractions: @kernel, @index

@inline exponential_profile(z, Lz, h) = (exp(z / h) - exp(-Lz / h)) / (1 - exp(-Lz / h))

function exponential_z_faces(; Nz, depth, h = Nz / 4.5)

    k = collect(1:Nz+1)
    z_faces = exponential_profile.(k, Nz, h)

    # Normalize
    z_faces .-= z_faces[1]
    z_faces .*= - depth / z_faces[end]

    z_faces[1] = 0.0

    return reverse(z_faces)
end

function simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo=(8, 8, 8))
    z = exponential_z_faces(; Nz, depth=4000) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        longitude = (0, 360), # Problem is here: when longitude is not periodic we get error
        latitude = (-60, -30), # Tentative southern ocean latitude range
        topology = (Periodic, Bounded, Bounded)
    )

    return grid
end

Nx = 362
Ny = 32
Nz = 30

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch  = ReactantState()
rgrid = simple_latitude_longitude_grid(rarch, Nx, Ny, Nz)

abstract type AbstractBadGrid{FT, Arch} end

struct BadGrid{FT, Arch, I} <: AbstractBadGrid{FT, Arch}
    architecture :: Arch
    Nx :: I
    Ny :: I
    Nz :: I
    Lx :: FT
end

Base.eltype(::AbstractBadGrid{FT}) where FT = FT

function BadGrid(architecture::Arch, Nx::I, Ny::I, Nz::I, Lx::FT) where {Arch, FT, I}
    return BadGrid{FT, Arch, I}(architecture, Nx, Ny, Nz, Lx)
end

function bad_wrapper!(u, arch, grid, Nx, Ny, Nz)

    workgroup = (16, 16)
    worksize  = (Nx, Ny, Nz)
    dev       = Oceananigans.Architectures.device(arch)
    loop!     = bad_kernel!(dev, workgroup, worksize)

    loop!(u, grid, Ny, Nz)

    return nothing
end

@kernel function bad_kernel!(u, grid, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    FT  = eltype(grid)
    d   = -1.0
    N²  = 3.0
    N²⁺ = max(zero(N²), N²)
    ℓ   = ifelse(N²⁺ == 0, FT(Inf), 1 / sqrt(N²⁺))
    ℓ   = min(d, ℓ)

    κu★ = ifelse(isnan(ℓ), d, ℓ)
    κu★ = FT(κu★)

    on_periphery    = (j < 1) | (j > Ny) | (k > Nz) | (k < 2)
    within_inactive = (j < 1) | (j > Ny) | (k < 1) | (k > Nz + 1)
    nan             = convert(FT, NaN)

    κu★ = ifelse(on_periphery, 0.0, ifelse(within_inactive, nan, κu★))

    @inbounds u[i, j, k] = κu★
end

bad_grid = BadGrid(rarch, Nx, Ny, Nz, 3.0)

u = zeros(Nx, Ny, Nz+1)

@info "Compiling..."
tic = time()
rbad_wrapper! = @compile raise_first=true raise=true sync=true bad_wrapper!(u, ReactantState(), rgrid, Nx, Ny, Nz)
compile_toc = time() - tic

@show compile_toc

@info "Running..."

rbad_wrapper!(u, ReactantState(), rgrid)
