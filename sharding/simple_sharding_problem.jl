# /home/avik-pal/.julia/bin/mpiexecjl -np 4 --project=. julia --threads=32 --color=yes --startup=no GB-25/sharding/simple_sharding_problem.jl

# mkpath("xla_dumps")
# tmpname = tempname("xla_dumps")

# ENV["CUDA_VISIBLE_DEVICES"] = ""
# ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
# ENV["XLA_FLAGS"] = " --xla_dump_to=xla_dumps/$(tmpname)/"
ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

# @info ENV["XLA_FLAGS"]

# using MPI
# MPI.Init()  # Only needed if using MPI to detect the coordinator

using Oceananigans
using Reactant

# using Cthulhu
using Dates

Reactant.Distributed.initialize()

ndevices = length(Reactant.devices())
nxdevices = floor(Int, sqrt(ndevices))
nydevices = ndevices ÷ nxdevices

process_id = Reactant.Distributed.local_rank()

arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition=Partition(nxdevices, nydevices, 1)
)

# arch = Oceananigans.ReactantState()

Nx = Ny = 16
Nz = 4

@info "[$(process_id)] creating tripolar grid" now(UTC)
grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=(0, 1))

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

@info "[$(process_id)] creating immersed boundary grid" now(UTC)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_islands))

free_surface = SplitExplicitFreeSurface(substeps=3)
model = HydrostaticFreeSurfaceModel(; grid, free_surface)
# model = HydrostaticFreeSurfaceModel(; grid)

model.clock.last_Δt = ConcreteRNumber(60.0)

@info "[$(process_id)] compiling first time step" now(UTC)
compiled_first_time_step! = @compile Oceananigans.TimeSteppers.first_time_step!(model, model.clock.last_Δt)
compiled_time_step! = @compile Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)

@info "[$(process_id)] running first time step" now(UTC)
@time "[$(process_id)] first time step" compiled_first_time_step!(model, model.clock.last_Δt)
@info "[$(process_id)] running second time step" now(UTC)
@time "[$(process_id)] second time step" compiled_time_step!(model, model.clock.last_Δt)
