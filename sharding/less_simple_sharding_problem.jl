# Unset environment variables which would cause XLA distributed to hang indefinitely.
for key in ("no_proxy", "http_proxy", "https_proxy", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY")
    delete!(ENV, key)
end

using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"
jobid_procid = string(get(ENV, "SLURM_JOB_ID", Int(datetime2unix(now(UTC)) * 1000)), ".", get(ENV, "SLURM_PROCID", string(getpid())))

using Oceananigans
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Reactant
using GordonBell25: GordonBell25

using Libdl: dllist

@show filter(contains("nccl"), dllist())

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
Reactant.Compiler.DUS_TO_CONCAT[] = true

GordonBell25.initialize(; single_gpu_per_process=false)

ndevices = length(Reactant.devices())

process_id = Reactant.Distributed.local_rank()
arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition=Partition(factors(ndevices)..., 1)
)

H = 8 # halo size
T = Tx, Ty = 512 .* factors(ndevices)
Nx, Ny = @. T - 2 * H
Nz = 256

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

@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

##### Latlong grid
@info "[$(process_id)] creating latlong grid" now(UTC)
grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(H, H, H), z=(-4000, 0),
                             latitude = (-80, 80),
                             longitude = (0, 360)
                             )

@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

free_surface = SplitExplicitFreeSurface(substeps=32)
momentum_advection = WENOVectorInvariant(order=5)
tracer_advection = WENO(order=5)
tracers = (:T, :S, :e)
equation_of_state = TEOS10EquationOfState()
buoyancy = SeawaterBuoyancy(; equation_of_state)
closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
model = HydrostaticFreeSurfaceModel(; grid, tracers, free_surface,
                                    momentum_advection, tracer_advection,
                                    buoyancy, closure)

@show size(parent(model.velocities.u))
# @assert size(parent(model.velocities.u)) == size(parent(model.tracers.T))
# @assert size(parent(model.velocities.v)) == size(parent(model.tracers.T))
# @assert size(parent(model.velocities.w)) == size(parent(model.tracers.T))

@show model

model.clock.last_Δt = ConcreteRNumber(60.0)

@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

# @info "[$(process_id)] compiling first time step" now(UTC)
# compiled_first_time_step! = @compile sync=true Oceananigans.TimeSteppers.first_time_step!(model, model.clock.last_Δt)
# @info "[$(process_id)] compiling second time step" now(UTC)
# compiled_time_step! = @compile sync=true Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)
# compiled_update_state! = @compile sync=true Oceananigans.TimeSteppers.update_state!(model)
# @info "[$(process_id)] allocations" GordonBell25.allocatorstats()

@info "[$(process_id)] compiling first time step" now(UTC)
compiled_first_time_step! = @compile sync=true raise=true GordonBell25.first_time_step!(model)
@info "[$(process_id)] compiling loop" now(UTC)
Ninner = ConcreteRNumber(128; sharding=Sharding.NamedSharding(arch.connectivity, ()))
compiled_loop! = @compile sync=true raise=true GordonBell25.loop!(model, Ninner)
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

# #-------------------------------------------------------------------------------
# code = @code_hlo optimize=:before_raise first_time_step!(model)
# open(joinpath(Reactant.MLIR.IR.DUMP_MLIR_DIR[], "first_time_step_before_raise_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# code = @code_hlo optimize=true first_time_step!(model)
# open(joinpath(Reactant.MLIR.IR.DUMP_MLIR_DIR[], "first_time_step_optimised_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# code = @code_hlo optimize=:before_raise loop!(model, Ninner)
# open(joinpath(Reactant.MLIR.IR.DUMP_MLIR_DIR[], "loop_before_raise_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# code = @code_hlo optimize=true loop!(model, Ninner)
# open(joinpath(Reactant.MLIR.IR.DUMP_MLIR_DIR[], "loop_optimised_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# #-------------------------------------------------------------------------------

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)
mkpath(profile_dir)

mkpath(joinpath(profile_dir, "first_time_step"))
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
@info "[$(process_id)] running first time step" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "first_time_step")) do
    @time "[$(process_id)] first time step" compiled_first_time_step!(model)
end

mkpath(joinpath(profile_dir, "loop"))
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
@info "[$(process_id)] running loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop")) do
    @time "[$(process_id)] loop" compiled_loop!(model, Ninner)
end

mkpath(joinpath(profile_dir, "loop2"))
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()
@info "[$(process_id)] running second loop" now(UTC)
@info "[$(process_id)] running loop2" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop2")) do
    @time "[$(process_id)] loop" compiled_loop!(model, Ninner)
end
@info "[$(process_id)] allocations" GordonBell25.allocatorstats()

@info "Done!" now(UTC)
