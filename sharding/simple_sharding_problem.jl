using GordonBell25
using Oceananigans
using Reactant
using Dates
using MPI
using CUDA

MPI.Init()  # Only needed if using MPI to detect the coordinator
Reactant.Distributed.initialize()

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"
arch_kind = get(ENV, "OCEANANIGANS_ARCHITECTURE", "ReactantState")
float_type = get(ENV, "FLOAT_TYPE", "Float64")

FT = if float_type == "Float64"
    Float64
elseif float_type == "Float32"
    Float32
end

Oceananigans.defaults.FloatType = FT

local_arch, Nd = if arch_kind == "ReactantState"
    Nd = length(Reactant.devices())
    (Oceananigans.ReactantState(), Nd)
else
    Nd = MPI.Comm_size(MPI.COMM_WORLD)
    (GPU(), Nd)
end

function similar_factors(N)
    N == 1 && return (1, 1)
    d = log2(N) / 2
    D = exp2(ceil(Int, d)) |> Int

    alternate = 1
    tries = 1
    while (N % D != 0)
        D -= tries * alternate
        tries += 1
        alternate *= -1
    end

    return D, N ÷ D
end

Rx, Ry = similar_factors(Nd)

if Nd == 1
    arch = local_arch
    rank = 0
else
    Rx = floor(Int, sqrt(Nd))
    partition = Partition(Rx, Ry, 1)
    arch = Oceananigans.Distributed(local_arch; partition)
    @show CUDA.device()
    rank = arch.local_rank
end

using Oceananigans.DistributedComputations: @handshake
@handshake @show arch

# Nx = 2048 * Rx
# Ny = 1024 * Ry
# Nz = 192

Nx = 1024 * Rx
Ny = 512 * Ry
Nz = 128

longitude = (0, 360)
latitude = (-80, 80)
z = (-1000, 0)
grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7),
                             longitude=(0, 360), latitude=(-80, 80), z=(-1000, 0))

# free_surface = ExplicitFreeSurface(gravitational_acceleration=1)
# model = HydrostaticFreeSurfaceModel(; grid, free_surface)

Δt = if local_arch isa Oceananigans.ReactantState
    ConcreteRNumber(FT(60.0))
else
    FT(60.0)
end

free_surface = SplitExplicitFreeSurface(substeps=30)
closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
model = GordonBell25.baroclinic_instability_model(grid; Δt, free_surface, closure)
@handshake @show model

# Prep time stepping
Nt = 100

if local_arch isa Oceananigans.ReactantState
    Nt = if arch isa Distributed
        replicated = Sharding.NamedSharding(arch.connectivity, ())
        ConcreteRNumber(Nt; sharding=replicated)
    else
        ConcreteRNumber(Nt)
    end

    @info "[$rank] compiling first time step" now(UTC)
    first! = @compile GordonBell25.first_time_step!(model)

    @info "[$rank] compiling second time step" now(UTC)
    step!   = @compile GordonBell25.time_step!(model)

    @info "[$rank] compiling loop" now(UTC)
    loop!   = @compile GordonBell25.loop!(model, Nt)
else
    loop! = GordonBell25.loop!
    step! = GordonBell25.time_step!
    first! = GordonBell25.first_time_step!
end

function untraced_loop!(model, Nt)
    for n = 1:Nt
        GordonBell25.time_step!(model)
    end
end

@info "[$rank] first"
CUDA.@time first!(model)
@info "[$rank] step"
CUDA.@time step!(model)
@info "[$rank] step"
CUDA.@time step!(model)
@info "[$rank] step"
CUDA.@time step!(model)

# @show CUDA.@profile step!(model)

sleep(3)

@info "[$rank] loop" 
CUDA.@time untraced_loop!(model, 10)
@info "[$rank] loop" 
CUDA.@time untraced_loop!(model, 10)
@info "[$rank] loop" 
CUDA.@time untraced_loop!(model, 10)

