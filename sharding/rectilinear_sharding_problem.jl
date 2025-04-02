using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans
using SeawaterPolynomials
using Reactant

using Libdl: dllist

@show filter(contains("nccl"), dllist())

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", string(ENV["SLURM_JOB_ID"], ".", ENV["SLURM_PROCID"]))
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true

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

function factors(N)
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

Nx, Ny = 32 .* factors(ndevices)
Nz = 32

@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

##### Latlong grid
@info "[$(process_id)] creating latlong grid" now(UTC)
grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7), z=(-4000, 0),
                       topology = (Periodic, Periodic, Bounded),
                       x = (0, 1e6),
                       y = (0, 1e6))

@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

momentum_advection = WENOVectorInvariant(order=5)
tracer_advection = WENO(order=5)
free_surface = ExplicitFreeSurface()
buoyancy = SeawaterBuoyancy(equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType))
model = HydrostaticFreeSurfaceModel(; grid, free_surface, momentum_advection, tracer_advection, buoyancy, tracers=(:T, :S))

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

@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

# @info "[$(process_id)] compiling first time step" now(UTC)
# compiled_first_time_step! = @compile sync=true Oceananigans.TimeSteppers.first_time_step!(model, model.clock.last_Δt)
# @info "[$(process_id)] compiling second time step" now(UTC)
# compiled_time_step! = @compile sync=true Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)
# compiled_update_state! = @compile sync=true Oceananigans.TimeSteppers.update_state!(model)
# @info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

@info "[$(process_id)] compiling first time step" now(UTC)
compiled_first_time_step! = @compile sync=true raise=true first_time_step!(model)
@info "[$(process_id)] compiling loop" now(UTC)
Ninner = ConcreteRNumber(2; sharding=Sharding.NamedSharding(arch.connectivity, ()))
compiled_loop! = @compile sync=true raise=true loop!(model, Ninner)
@info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

# #-------------------------------------------------------------------------------
# code = @code_hlo optimize=:before_raise first_time_step!(model)
# open(joinpath(@__DIR__, "first_time_step_before_raise_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# code = @code_hlo optimize=true first_time_step!(model)
# open(joinpath(@__DIR__, "first_time_step_optimised_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# code = @code_hlo optimize=:before_raise loop!(model, Ninner)
# open(joinpath(@__DIR__, "loop_before_raise_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# code = @code_hlo optimize=true loop!(model, Ninner)
# open(joinpath(@__DIR__, "loop_optimised_$(process_id).mlir"), "w") do io
#     show(IOContext(io, :debug => true), code)
# end
# #-------------------------------------------------------------------------------

profile_dir = joinpath(@__DIR__, "profiling", string(ENV["SLURM_JOB_ID"], ".", ENV["SLURM_PROCID"]))
mkpath(profile_dir)
Reactant.with_profiler(profile_dir) do

    # @info "[$(process_id)] running first time step" now(UTC)
    # @info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()
    # @time "[$(process_id)] first time step" compiled_first_time_step!(model, model.clock.last_Δt)
    # @info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()
    # @info "[$(process_id)] running loop" now(UTC)
    # @time "[$(process_id)] loop" for iter in 2:10
    #     @info "[$(process_id)] iterating" iter now(UTC)
    #     @time "[$(process_id)] $(iter)-th timestep" compiled_update_state!(model)
    #     if iter < 10 || iszero(iter % 50)
    #         @info "[$(process_id)] allocations" iter Reactant.XLA.allocatorstats()
    #     end
    # end

    Ninner = ConcreteRNumber(10; sharding=Sharding.NamedSharding(arch.connectivity, ()))
    @info "[$(process_id)] running first time step" now(UTC)
    @info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()
    @time "[$(process_id)] first time step" compiled_first_time_step!(model)
    @info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()
    @info "[$(process_id)] running loop" now(UTC)
    @time "[$(process_id)] loop" compiled_loop!(model, Ninner)
    @info "[$(process_id)] allocations" Reactant.XLA.allocatorstats()

end

@info "Done!" now(UTC)
