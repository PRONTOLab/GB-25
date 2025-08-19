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
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Grids: static_column_depthᶜᶜᵃ, peripheral_node, inactive_node
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: compute_diffusivities!, getclosure, depthᶜᶜᶠ, height_above_bottomᶜᶜᶠ
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: compute_CATKE_diffusivities!, mask_diffusivity, κuᶜᶜᶠ, κcᶜᶜᶠ, κeᶜᶜᶠ, momentum_mixing_lengthᶜᶜᶠ, convective_length_scaleᶜᶜᶠ, turbulent_velocityᶜᶜᶜ, stability_functionᶜᶜᶠ, stable_length_scaleᶜᶜᶠ, stratification_mixing_lengthᶜᶜᶠ
using Oceananigans.Utils: launch!, _launch!, configure_kernel, work_layout

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

function bad_wrapper!(u, arch, grid; parameters = :xyz)
    bad_launch!(arch, grid, parameters,
            bad_kernel!,
            u, grid)

    return nothing
end

# Inner interface
@inline function bad_launch!(arch, grid, workspec, kernel!, first_kernel_arg, other_kernel_args...;
                          exclude_periphery = false,
                          reduced_dimensions = (),
                          active_cells_map = nothing)

    location = (Nothing, Nothing, Nothing)

    loop!, worksize = bad_configure_kernel(arch, grid, workspec, kernel!, active_cells_map, Val(exclude_periphery);
                                       location,
                                       reduced_dimensions)

    loop!(first_kernel_arg, other_kernel_args...)

    return nothing
end

@inline function bad_configure_kernel(arch, grid, workspec, kernel!, ::Nothing, args...; 
                                  reduced_dimensions = (),
                                  location = nothing)

    workgroup, worksize = work_layout(grid, workspec, reduced_dimensions)
    dev  = Oceananigans.Architectures.device(arch)
    loop = kernel!(dev, workgroup, worksize)

    return loop, worksize
end

@kernel function bad_kernel!(u, grid)
    i, j, k = @index(Global, NTuple)

    FT  = eltype(grid)
    d   = -1.0
    N²  = 3.0
    N²⁺ = max(zero(N²), N²)
    ℓ   = ifelse(N²⁺ == 0, FT(Inf), 1 / sqrt(N²⁺))
    ℓ   = min(d, ℓ)

    κu★ = ifelse(isnan(ℓ), d, ℓ)
    κu★ = FT(κu★)

    on_periphery    = (j < 1) | (j > grid.Ny) | (k > grid.Nz) | (k < 2)
    within_inactive = (j < 1) | (j > grid.Ny) | (k < 1) | (k > grid.Nz + 1)
    nan             = convert(FT, NaN)

    κu★ = ifelse(on_periphery, 0.0, ifelse(within_inactive, nan, κu★))

    @inbounds u[i, j, k] = κu★
end

bad_grid = BadGrid(rarch, Nx, Ny, Nz, 3.0)

u = zeros(Nx, Ny, Nz+1)

@show @which configure_kernel(ReactantState(), rgrid, :xyz, bad_kernel!, nothing, Val(false); location=(Nothing, Nothing, Nothing), reduced_dimensions=())

@info "Compiling..."
tic = time()
rbad_wrapper! = @compile raise_first=true raise=true sync=true bad_wrapper!(u, ReactantState(), rgrid; parameters = :xyz)
compile_toc = time() - tic

@show compile_toc

@info "Running..."

rbad_wrapper!(u, ReactantState(), rgrid; parameters = :xyz)
