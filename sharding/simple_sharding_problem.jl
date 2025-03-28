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

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", string(ENV["SLURM_JOB_ID"], ".", ENV["SLURM_PROCID"]))
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true

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

Nx = Ny = 512 * ndevices
Nz = 128

#=
##### Tripolar Grid
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
=#

@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

##### Latlong grid
@info "[$(process_id)] creating latlong grid" now(UTC)
grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7), z=(-4000, 0),
                             latitude = (-80, 80),
                             longitude = (0, 360)
                             )

@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

free_surface = ExplicitFreeSurface()
model = HydrostaticFreeSurfaceModel(; grid, free_surface)
# model = HydrostaticFreeSurfaceModel(; grid)

@show model

model.clock.last_Δt = ConcreteRNumber(60.0)

function first_time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

function loop!(model, Ninner)
    Δt = model.clock.last_Δt
    @trace track_numbers=false for _ = 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

Ninner = ConcreteRNumber(2)

@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

@info "[$(process_id)] compiling first time step" now(UTC)
compiled_first_time_step! = @compile Oceananigans.TimeSteppers.first_time_step!(model, model.clock.last_Δt)
@info "[$(process_id)] compiling second time step" now(UTC)
compiled_time_step! = @compile Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)
compiled_update_state! = @compile Oceananigans.TimeSteppers.update_state!(model)

@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

# @info "[$(process_id)] compiling first time step" now(UTC)
# compiled_first_time_step! = @compile first_time_step!(model)
# @info "[$(process_id)] compiling second time step" now(UTC)
# compiled_loop! = @compile loop!(model, Ninner)

# code = @code_hlo optimize=:before_raise Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)
# open(joinpath(@__DIR__, "module_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end

@info "[$(process_id)] running first time step" now(UTC)
@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()
@time "[$(process_id)] first time step" compiled_first_time_step!(model, model.clock.last_Δt)
@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()
@info "[$(process_id)] running loop" now(UTC)
@time "[$(process_id)] loop" for iter in 2:1000
    @info "[$(process_id)] iterating" iter now(UTC)
    @time "[$(process_id)] $(iter)-th timestep" compiled_update_state!(model)
    if iter < 10 || iszero(iter % 50)
        @info "[$(process_id)] allocations" iter Reactant.XLA.allocatorstats()
    end
    #GC.gc(true); GC.gc(false); GC.gc(true)
end

# Ninner = ConcreteRNumber(100)
# @info "[$(process_id)] running first time step" now(UTC)
# @time "[$(process_id)] first time step" compiled_first_time_step!(model)
# @info "[$(process_id)] running loop" now(UTC)
# @time "[$(process_id)] loop" compiled_loop!(model, Ninner)

@info "Done!" now(UTC)
