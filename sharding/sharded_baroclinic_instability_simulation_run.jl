using Dates
@info "This is when the fun begins" now(UTC)

using ArgParse

const args_settings = ArgParseSettings()
@add_arg_table! args_settings begin
    "--grid-x"
        help = "Base factor for number of grid points on the x axis."
        default = 1536
        arg_type = Int
    "--grid-y"
        help = "Base factor for number of grid points on the y axis."
        default = 768
        arg_type = Int
    "--test-type"
        help = "Which test to run (fill_all_halos, fill_east_west, fill_north_south)."
        default = "fill_all_halos"
        arg_type = String
    "--grid-z"
        help = "Base factor for number of grid points on the z axis."
        default = 4
        arg_type = Int
end
const parsed_args = parse_args(ARGS, args_settings)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: fill_one_halo!, first_time_step!, time_step!, loop!, factors, is_distributed_env_present
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

jobid_procid = GordonBell25.get_jobid_procid()

# This must be called before `GordonBell25.initialize`!
GordonBell25.preamble()

using Libdl: dllist
@show filter(contains("nccl"), dllist())

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
# Reactant.Compiler.DEBUG_PRINT_CODEGEN[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
# Reactant.Compiler.DUS_TO_CONCAT[] = false
# Reactant.Compiler.SUM_TO_REDUCEWINDOW[] = true
# Reactant.Compiler.AGGRESSIVE_SUM_TO_CONV[] = true

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

Rx, Ry = factors(Ndev)
if Ndev == 1
    rank = 0
    arch = Oceananigans.ReactantState()
else
    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(Rx, Ry, 1)
    )
    rank = Reactant.Distributed.local_rank()
end

@info "[$rank] allocations" GordonBell25.allocatorstats()
H = 8
Tx = parsed_args["grid-x"] * Rx
Ty = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nx = Tx - 2H
Ny = Ty - 2H

@info "[$rank] Generating model..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@info "[$rank] allocations" GordonBell25.allocatorstats()

@show model

Ninner = ConcreteRNumber(256; sharding=Sharding.NamedSharding(arch.connectivity, ()))


# @info "[$rank] Compiling first_time_step!..." now(UTC)
# rfirst! = @compile sync=true raise=true first_time_step!(model)
# @info "[$rank] allocations" GordonBell25.allocatorstats()

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)
test_type = parsed_args["test-type"]

if test_type == "fill_all_halos"
    @info "[$rank] Compiling fill_halo_regions!..." now(UTC)
    rfill_all_halos!= @compile sync=true raise=true fill_halo_regions!(model.tracers.T)
    @info "[$rank] allocations" GordonBell25.allocatorstats()

    mkpath(joinpath(profile_dir, "fill_all_halos"))
    @info "[$rank] allocations" GordonBell25.allocatorstats()
    @info "[$rank] Running test code..." now(UTC)
    Reactant.with_profiler(joinpath(profile_dir, "fill_all_halos")) do
        @time "[$rank] test " rfill_all_halos!(model.tracers.T)
    end
    @info "[$rank] allocations" GordonBell25.allocatorstats()

elseif test_type == "fill_north_south"

    loc = (Center(), Center(), Center())
    north_south_bcs = FieldBoundaryConditions(model.grid, loc, west=nothing, east=nothing, top=nothing, bottom=nothing)
    north_south_field = CenterField(model.grid; boundary_conditions=north_south_bcs)

    @info "[$rank] Compiling fill_halo_regions! for east_west_field..." now(UTC)
    rfill_east_west! = @compile sync=true raise=true fill_halo_regions!(east_west_field)
    @info "[$rank] allocations" GordonBell25.allocatorstats()

    mkpath(joinpath(profile_dir, "fill_east_west"))
    @info "[$rank] allocations" GordonBell25.allocatorstats()
    @info "[$rank] Running test code..." now(UTC)
    Reactant.with_profiler(joinpath(profile_dir, "fill_east_west")) do
        @time "[$rank] test " rfill_east_west!(east_west_field)
    end
    @info "[$rank] allocations" GordonBell25.allocatorstats()

elseif test_type == "fill_east_west"

    loc = (Center(), Center(), Center())
    east_west_bcs = FieldBoundaryConditions(model.grid, loc, south=nothing, north=nothing, top=nothing, bottom=nothing)
    east_west_field = CenterField(model.grid; boundary_conditions=east_west_bcs)

    @info "[$rank] Compiling fill_halo_regions! for north_south_field..." now(UTC)
    rfill_north_south! = @compile sync=true raise=true fill_halo_regions!(north_south_field)
    @info "[$rank] allocations" GordonBell25.allocatorstats()

    mkpath(joinpath(profile_dir, "fill_north_south"))
    @info "[$rank] allocations" GordonBell25.allocatorstats()
    @info "[$rank] Running test code..." now(UTC)
    Reactant.with_profiler(joinpath(profile_dir, "fill_north_south")) do
        @time "[$rank] test " rfill_north_south!(north_south_field)
    end
    @info "[$rank] allocations" GordonBell25.allocatorstats()
end

@info "Done!" now(UTC)
