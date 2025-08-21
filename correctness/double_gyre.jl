ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans.Grids: LatitudeLongitudeGrid, Periodic, Bounded, AbstractHorizontallyCurvilinearGrid
using Oceananigans.Architectures: ReactantState, device
using Reactant
using KernelAbstractions
using GordonBell25
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)
using OffsetArrays

using InteractiveUtils
using KernelAbstractions: @kernel, @index

using Oceananigans: Distributed, DistributedComputations, CPU,
                    OrthogonalSphericalShellGrids
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: RightConnected
#=
abstract type AbstractGrid{FT, TX, TY, TZ, Arch} end
abstract type AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end
abstract type AbstractCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch} end
abstract type AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch} <: AbstractCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch} end

Base.eltype(::AbstractGrid{FT}) where FT = FT
=#

struct BadLatitudeLongitudeGrid{FT, TX, TY, TZ, Z, DXF, DXC, XF, XC, DYF, DYC, YF, YC,
                             DXCC, DXFC, DXCF, DXFF, DYFC, DYCF, Arch, I} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Z, Arch}
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


Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{BadLatitudeLongitudeGrid{FT, TX, TY, TZ, Z, DXF, DXC, XF, XC, DYF, DYC, YF, YC, 
                                                 DXCC, DXFC, DXCF, DXFF, DYFC, DYCF, Arch, I}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, Z, DXF, DXC, XF, XC, DYF, DYC, YF, YC, DXCC, DXFC, DXCF, DXFF, DYFC, DYCF, Arch, I} 
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    Z2 = Reactant.traced_type_inner(Z, seen, mode, track_numbers, sharding, runtime)
    DXF2 = Reactant.traced_type_inner(DXF, seen, mode, track_numbers, sharding, runtime)
    DXC2 = Reactant.traced_type_inner(DXC, seen, mode, track_numbers, sharding, runtime)
    XF2 = Reactant.traced_type_inner(XF, seen, mode, track_numbers, sharding, runtime)
    XC2 = Reactant.traced_type_inner(XC, seen, mode, track_numbers, sharding, runtime)
    DYF2 = Reactant.traced_type_inner(DYF, seen, mode, track_numbers, sharding, runtime)
    DYC2 = Reactant.traced_type_inner(DYC, seen, mode, track_numbers, sharding, runtime)
    YF2 = Reactant.traced_type_inner(YF, seen, mode, track_numbers, sharding, runtime)
    YC2 = Reactant.traced_type_inner(YC, seen, mode, track_numbers, sharding, runtime)
    DXCC2 = Reactant.traced_type_inner(DXCC, seen, mode, track_numbers, sharding, runtime)
    DXFC2 = Reactant.traced_type_inner(DXFC, seen, mode, track_numbers, sharding, runtime)
    DXCF2 = Reactant.traced_type_inner(DXCF, seen, mode, track_numbers, sharding, runtime)
    DXFF2 = Reactant.traced_type_inner(DXFF, seen, mode, track_numbers, sharding, runtime)
    DYFC2 = Reactant.traced_type_inner(DYFC, seen, mode, track_numbers, sharding, runtime)
    DYCF2 = Reactant.traced_type_inner(DYCF, seen, mode, track_numbers, sharding, runtime)
    I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)

    FT2 = Reactant.traced_type_inner(FT, seen, mode, track_numbers, sharding, runtime)

    for NF in (XF2, XC2, YF2, YC2, DXCC2, DXFC2, DYCF2, DYCF2, DXFF2)
	if NF === Nothing
	   continue
	end
	FT2 = Reactant.promote_traced_type(FT2, eltype(NF))
    end

    res = BadLatitudeLongitudeGrid{FT2, TX2, TY2, TZ2, Z2, DXF2, DXC2, XF2, XC2, DYF2, DYC2, YF2, YC2, 
                                                 DXCC2, DXFC2, DXCF2, DXFF2, DYFC2, DYCF2, Arch, I2}
    return res
end


@inline Reactant.make_tracer(
    seen,
    @nospecialize(prev::BadLatitudeLongitudeGrid),
    args...;
    kwargs...
    ) = Reactant.make_tracer_via_immutable_constructor(seen, prev, args...; kwargs...)

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

    # NEED TO MAKE THIS CONSTRUCT A BadLatitudeLongitudeGrid struct instead
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

    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ = bad_allocate_metrics(architecture)

    grid = BadLatitudeLongitudeGrid{TX, TY, TZ}(architecture,
                                                         Nλ, Nφ, Nz,
                                                         Hλ, Hφ, Hz,
                                                         Lλ, Lφ, Lz,
                                                         Δλᶠᵃᵃ, Δλᶜᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ,
                                                         Δφᵃᶠᵃ, Δφᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ,
                                                         z,
                                                         Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ,
                                                         Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
    
    return grid
end

function bad_allocate_metrics(arch)
    FT = Float64

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

function simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo=(8, 8, 8))
    z = Array(range(-4000, 0, Nz))

    grid = BadLatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        topology = (Periodic, Bounded, Bounded)
    )

    return grid
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
u = Reactant.ConcreteRArray(u)

@show @which eltype(rgrid)

@info "Compiling..."
tic = time()
rbad_wrapper! = @compile raise_first=true raise=true sync=true bad_wrapper!(u, rarch, rgrid, Nx, Ny, Nz)
compile_toc = time() - tic

@show compile_toc

@info "Running..."

rbad_wrapper!(u, rarch, rgrid, Nx, Ny, Nz)
