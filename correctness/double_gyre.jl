ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans.Grids: LatitudeLongitudeGrid, Periodic, Bounded
using Oceananigans.Architectures: ReactantState, device
using Reactant
using KernelAbstractions
using GordonBell25
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)
using OffsetArrays

using InteractiveUtils
using KernelAbstractions: @kernel, @index

abstract type AbstractBadGrid{FT, Arch} end

Base.eltype(::AbstractBadGrid{FT}) where FT = FT

struct BadLatitudeLongitudeGrid{FT, TX, TY, TZ, Z, DXF, DXC, XF, XC, DYF, DYC, YF, YC,
                             DXCC, DXFC, DXCF, DXFF, DYFC, DYCF, Arch, I} <: AbstractBadGrid{FT, Arch}
    architecture :: Arch
    Nx :: I
    Ny :: I
    Nz :: I
    Hx :: I
    Hy :: I
    Hz :: I
    Lx :: FT
    Ly :: FT
    Lz :: FT
    # All directions can be either regular (FX, FY, FZ) <: Number
    # or stretched (FX, FY, FZ) <: AbstractVector
    Δλᶠᵃᵃ :: DXF
    Δλᶜᵃᵃ :: DXC
    λᶠᵃᵃ  :: XF
    λᶜᵃᵃ  :: XC
    Δφᵃᶠᵃ :: DYF
    Δφᵃᶜᵃ :: DYC
    φᵃᶠᵃ  :: YF
    φᵃᶜᵃ  :: YC
    z     :: Z
    # Precomputed metrics M <: Nothing means metrics will be computed on the fly
    Δxᶜᶜᵃ :: DXCC
    Δxᶠᶜᵃ :: DXFC
    Δxᶜᶠᵃ :: DXCF
    Δxᶠᶠᵃ :: DXFF
    Δyᶠᶜᵃ :: DYFC
    Δyᶜᶠᵃ :: DYCF
    Azᶜᶜᵃ :: DXCC
    Azᶠᶜᵃ :: DXFC
    Azᶜᶠᵃ :: DXCF
    Azᶠᶠᵃ :: DXFF
    # Spherical radius
    radius :: FT
end

function BadLatitudeLongitudeGrid{TX, TY, TZ}(architecture::Arch,
                                           Nλ::I, Nφ::I, Nz::I, Hλ::I, Hφ::I, Hz::I,
                                           Lλ :: FT, Lφ :: FT, Lz :: FT,
                                           Δλᶠᵃᵃ :: DXF, Δλᶜᵃᵃ :: DXC,
                                            λᶠᵃᵃ :: XF,   λᶜᵃᵃ :: XC,
                                           Δφᵃᶠᵃ :: DYF, Δφᵃᶜᵃ :: DYC,
                                            φᵃᶠᵃ :: YF,   φᵃᶜᵃ :: YC, z :: Z,
                                           Δxᶜᶜᵃ :: DXCC, Δxᶠᶜᵃ :: DXFC,
                                           Δxᶜᶠᵃ :: DXCF, Δxᶠᶠᵃ :: DXFF,
                                           Δyᶠᶜᵃ :: DYFC, Δyᶜᶠᵃ :: DYCF,
                                           Azᶜᶜᵃ :: DXCC, Azᶠᶜᵃ :: DXFC,
                                           Azᶜᶠᵃ :: DXCF, Azᶠᶠᵃ :: DXFF,
                                           radius :: FT) where {Arch, FT, TX, TY, TZ, Z,
                                                                DXF, DXC, XF, XC,
                                                                DYF, DYC, YF, YC,
                                                                DXFC, DXCF,
                                                                DXFF, DXCC,
                                                                DYFC, DYCF, I}

    return LatitudeLongitudeGrid{FT, TX, TY, TZ, Z,
                                 DXF, DXC, XF, XC,
                                 DYF, DYC, YF, YC,
                                 DXCC, DXFC, DXCF, DXFF,
                                 DYFC, DYCF, Arch, I}(architecture,
                                                      Nλ, Nφ, Nz,
                                                      Hλ, Hφ, Hz,
                                                      Lλ, Lφ, Lz,
                                                      Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                      Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, z,
                                                      Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                      Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                                                      Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

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

    grid = BadLatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        topology = (Periodic, Bounded, Bounded)
    )

    return grid
end

const R_Earth = 6371.0e3

function BadLatitudeLongitudeGrid(architecture = CPU(),
                               FT::DataType = Float64;
                               size,
                               z = nothing,
                               radius = R_Earth,
                               topology = nothing,
                               halo = nothing)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    # Calculate all direction (which might be stretched)
    # A direction is regular if the domain passed is a Tuple{<:Real, <:Real},
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    TX, TY, TZ = topology

    Lλ = 360.0
    Lφ = 30.0
    Lz = 4000.0

    λᶠᵃᵃ  = zeros(377)
    λᶜᵃᵃ  = zeros(377)
    Δλᶠᵃᵃ = 0.99
    Δλᶜᵃᵃ = 0.99

    φᵃᶠᵃ  = zeros(48)
    φᵃᶜᵃ  = zeros(47)
    Δφᵃᶠᵃ = 0.99
    Δφᵃᶜᵃ = 0.99

    grid = BadLatitudeLongitudeGrid{TX, TY, TZ}(architecture,
                                                         Nλ, Nφ, Nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         z,
                                                         (nothing for i=1:10)..., FT(radius))

    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = bad_allocate_metrics(grid)

    grid = BadLatitudeLongitudeGrid{TX, TY, TZ}(grid.architecture,
                                             Nλ, Nφ, Nz,
                                             Hλ, Hφ, Hz,
                                             grid.Lx, grid.Ly, grid.Lz,
                                             grid.Δλᶠᵃᵃ, grid.Δλᶜᵃᵃ, grid.λᶠᵃᵃ, grid.λᶜᵃᵃ,
                                             grid.Δφᵃᶠᵃ, grid.Δφᵃᶜᵃ, grid.φᵃᶠᵃ, grid.φᵃᶜᵃ,
                                             grid.z,
                                             Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                                             Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, grid.radius)

    return grid
end

function bad_allocate_metrics(grid)
    FT = Float64
    arch = grid.architecture

    offsets     = (2,2)
    metric_size = (8,8)

    Δxᶜᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Δxᶠᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Δxᶜᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Δxᶠᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶜᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶠᶜ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶜᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)
    Azᶠᶠ = OffsetArray(zeros(arch, FT, metric_size...), offsets...)

    Δyᶠᶜ = zeros(1, 1, 1)
    Δyᶜᶠ = zeros(1, 1, 1)

    return Δxᶜᶜ, Δxᶠᶜ, Δxᶜᶠ, Δxᶠᶠ, Δyᶠᶜ, Δyᶜᶠ, Azᶜᶜ, Azᶠᶜ, Azᶜᶠ, Azᶠᶠ
end

Nx = 362
Ny = 32
Nz = 30

@info "Generating model..."
rarch  = ReactantState()
rgrid = simple_latitude_longitude_grid(rarch, Nx, Ny, Nz)

function bad_wrapper!(u, arch, grid, Nx, Ny, Nz)

    workgroup = (16, 16)
    worksize  = (Nx, Ny, Nz)
    dev       = device(arch)
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

u = zeros(Nx, Ny, Nz+1)

@show @which eltype(rgrid)

z = exponential_z_faces(; Nz, depth=4000)
@show @which LatitudeLongitudeGrid(rarch; size=(Nx, Ny, Nz), halo=(8, 8, 8), z, longitude = (0, 360), latitude = (-60, -30), topology = (Periodic, Bounded, Bounded))

@info "Compiling..."
tic = time()
rbad_wrapper! = @compile raise_first=true raise=true sync=true bad_wrapper!(u, rarch, rgrid, Nx, Ny, Nz)
compile_toc = time() - tic

@show compile_toc

@info "Running..."

rbad_wrapper!(u, rarch, rgrid, Nx, Ny, Nz)
