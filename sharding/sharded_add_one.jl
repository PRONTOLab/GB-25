using Reactant
using KernelAbstractions: @kernel, @index
using Oceananigans

function add_dt!(u, Δt)
    grid = u.grid
    Oceananigans.Utils.launch!(grid.architecture, grid, :xyz, _add_dt!, u, Δt)
    return nothing
end

@kernel function _add_dt!(u, Δt)
    i, j, k = @index(Global, NTuple)

    FT = eltype(u)
    FT_Δt = convert(FT, Δt)
    @inbounds begin
        u[i, j, k] += FT_Δt
        # u[i, j, k] += 1 # also doesn't work
    end
end

# Use with XLA_FLAGS="--xla_force_host_platform_device_count=4"
arch = Oceananigans.Distributed(
    Oceananigans.Architectures.ReactantState(),
    partition = Partition(2, 2, 1)
)

rank = arch.local_rank

Nx = 96
Ny = 48
Nz = 4
longitude = (0, 360)
latitude = (-80, 80)
z = (0, 1)
grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), longitude, latitude, z)
u = XFaceField(grid)

add_dt_xla = @code_xla raise=true add_dt!(u, 1)

open("sharded_add_dt_4.xla", "w") do io
    print(io, add_dt_xla)
end
