using Reactant

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using SeawaterPolynomials

using NumericalEarth
using NumericalEarth.EarthSystemModels.InterfaceComputations: TenUnrolledIterations, ComponentInterfaces

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
    z = ExponentialDiscretization(Nz, -4000, 0; scale=1000)

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

# ── Moist baroclinic wave IC kernels ──────────────────────────────────────

using Oceananigans.Grids: λnode, φnode, znode
using Breeze.AtmosphereModels: dynamics_density, specific_prognostic_moisture

@kernel function _set_moist_baroclinic_wave_kernel!(θ_field, ρ_field, qv_field, grid)
    i, j, k = @index(Global, NTuple)
    λ_deg = λnode(i, j, k, grid, Center(), Center(), Center())
    φ_deg = φnode(i, j, k, grid, Center(), Center(), Center())
    z     = znode(i, j, k, grid, Center(), Center(), Center())
    @inbounds begin
        θ_field[i, j, k] = initial_theta(λ_deg, φ_deg, z)
        ρ_field[i, j, k] = initial_density(λ_deg, φ_deg, z)
        qv_field[i, j, k] = initial_moisture(λ_deg, φ_deg, z)
    end
end

@kernel function _set_zonal_wind_kernel!(u_field, grid)
    i, j, k = @index(Global, NTuple)
    λ_deg = λnode(i, j, k, grid, Face(), Center(), Center())
    φ_deg = φnode(i, j, k, grid, Face(), Center(), Center())
    z     = znode(i, j, k, grid, Face(), Center(), Center())
    @inbounds u_field[i, j, k] = initial_zonal_wind(λ_deg, φ_deg, z)
end

function _set_moist_baroclinic_wave!(model)
    grid = model.grid
    arch = grid.architecture

    ρ  = dynamics_density(model.dynamics)
    θ  = model.temperature
    qv = specific_prognostic_moisture(model)
    u  = model.velocities.u

    Oceananigans.Utils.launch!(arch, grid, :xyz,
        _set_moist_baroclinic_wave_kernel!, θ, ρ, qv, grid)

    Oceananigans.Utils.launch!(arch, grid, :xyz,
        _set_zonal_wind_kernel!, u, grid)

    ρu = model.momentum.ρu
    parent(ρu) .= parent(ρ) .* parent(u)

    ρqv = model.moisture_density
    parent(ρqv) .= parent(ρ) .* parent(qv)

    return nothing
end

function set_moist_baroclinic_wave!(model)
    if model.grid.architecture isa ReactantState
        rset! = @compile sync=true raise=true _set_moist_baroclinic_wave!(model)
        rset!(model)
    else
        _set_moist_baroclinic_wave!(model)
    end
end

# ──────────────────────────────────────────────────────────────────────────

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, resolution, Nz)
    Nx, Ny = resolution_to_points(resolution)
    return gaussian_islands_tripolar_grid(arch, Nx, Ny, Nz)
end

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, Nx, Ny, Nz; halo=(8, 8, 8))
    # Grid setup
    z = ExponentialDiscretization(Nz, -4000, 0; scale=1000)
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo, z)

    zb = z[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

    return @gbprofile "ImmersedBoundaryGrid" ImmersedBoundaryGrid(underlying_grid,
                                                                  GridFittedBottom(gaussian_islands);
                                                                  active_cells_map = false)
end

