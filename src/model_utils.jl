using Reactant

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using SeawaterPolynomials

using ClimaOcean
using ClimaOcean.OceanSeaIceModels.InterfaceComputations: FixedIterations, ComponentInterfaces

using Dates
using Printf
using Profile
using Random
using Serialization

using KernelAbstractions: @index, @kernel

const PROFILE = Ref(false)

macro gbprofile(name::String, expr::Expr)
    return quote
        if $(PROFILE)[]
            $(Profile.clear)()
            $(Profile.init)(; delay=0.1)
            out = $(Profile).@profile $(esc(expr))
            open(string("profile_", $(esc(name)), ".txt"), "w") do s
                println(s, "# Showing profile of")
                println(s, "#     ", $(string(expr)))
                println(s, "# at ", $(string(__source__)))
                $(Profile.print)(IOContext(s, :displaysize => (48, 1000)))
            end
            $(Serialization.serialize)(string("profile_", $(esc(name)), ".dat"), $(Profile).retrieve())
            $(Profile.clear)()
            out
        else
            $(esc(expr))
        end
    end
end

function resolution_to_points(resolution)
    Nx = convert(Int, 384 / resolution)
    Ny = convert(Int, 192 / resolution)
    return Nx, Ny
end

function simple_latitude_longitude_grid(arch, resolution, Nz)
    Nx, Ny = resolution_to_points(resolution)
    return simple_latitude_longitude_grid(arch, Nx, Ny, Nz)
end

function simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo=(8, 8, 8))
    z = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        latitude = (-80, 80),
        longitude = (0, 360)
    )

    return grid
end

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

# Simple initial condition for producing pretty pictures
@inline function smooth_step(φ)
    φ₀ = 40
    dφ = 5
    return (1 - tanh((abs(φ) - φ₀) / dφ)) / 2
end

function Tᵢ(λ, φ, z)
    dTdz = 1e-3
    return (30 + dTdz * z) * smooth_step(φ) + rand()
end

function Sᵢ(λ, φ, z)
    dSdz = - 5e-3
    return dSdz * z + rand()
end

@kernel function set_baroclinic_instability_kernel!(T, S, grid)
    i, j, k = @index(Global, NTuple)
    φ = Oceananigans.Grids.φnode(i, j, k, grid, Center(), Center(), Center())
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    @inbounds begin
        dTdz = 1e-3
        T[i, j, k] = (30 + dTdz * z) * smooth_step(φ)

        dSdz = - 5e-3
        S[i, j, k] = dSdz * z
    end
end

function _set_baroclinic_instability!(model)
    grid = model.grid
    arch = grid.architecture
    Oceananigans.Utils.launch!(arch, grid, :xyz, set_baroclinic_instability_kernel!,
                               model.tracers.T, model.tracers.S, model.grid)
    return nothing
end

function set_baroclinic_instability!(model)
    if model.architecture isa ReactantState
        rset! = @compile sync=true raise=true _set_baroclinic_instability!(model)
        rset!(model)
    else
        _set_baroclinic_instability!(model)
    end
end

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, resolution, Nz)
    Nx, Ny = resolution_to_points(resolution)
    return gaussian_islands_tripolar_grid(arch, Nx, Ny, Nz)
end

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, Nx, Ny, Nz; halo=(8, 8, 8))
    # Grid setup
    z = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo, z)

    zb = z[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

    return @gbprofile "ImmersedBoundaryGrid" ImmersedBoundaryGrid(underlying_grid,
                                                                  GridFittedBottom(gaussian_islands);
                                                                  active_cells_map = false)
end

