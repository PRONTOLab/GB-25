using Dates
@info "This is when the fun begins" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, factors, is_distributed_env_present

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

jobid_procid = GordonBell25.get_jobid_procid()

GordonBell25.preamble()

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
# Resharding must remain enabled — the constructor sets! host arrays
# into sharded fields (bathymetry, T, S, atmosphere), which is a
# NoSharding -> DimsSharding reshard. The data-free baroclinic test
# disables resharding because it uses purely analytic init; we cannot.
Reactant.Compiler.WHILE_CONCAT[] = true

GordonBell25.initialize(; single_gpu_per_process=false)

devarch = Oceananigans.ReactantState()
arch    = devarch

Ndev = if arch isa Oceananigans.ReactantState
    length(Reactant.devices())
else
    comm = MPI.COMM_WORLD
    MPI.Comm_size(comm)
end

Rx, Ry = factors(Ndev)

if Ndev == 1
    rank = 0
else
    arch = Oceananigans.Distributed(devarch; partition = Partition(Rx, Ry, 1))
    rank = Reactant.Distributed.local_rank()
end

resolution = parse(Float64, get(ENV, "RESOLUTION", "2"))
Nz         = parse(Int,     get(ENV, "NZ",         "20"))
Δt         = parse(Float64, get(ENV, "DT_SECONDS", "30"))

@info "[$rank] Generating model (resolution=$resolution, Nz=$Nz)..." now(UTC)
model = GordonBell25.ocean_climate_model_init(arch;
                                              resolution = resolution,
                                              Nz         = Nz,
                                              Δt         = Δt)

Ninner = 256

if devarch isa Oceananigans.ReactantState
    Ninner = if Ndev == 1
        ConcreteRNumber(Ninner)
    else
        sharding = Sharding.NamedSharding(arch.connectivity, ())
        ConcreteRNumber(Ninner; sharding)
    end
end

@info "[$rank] Compiling first_time_step!..." now(UTC)
compile_options = CompileOptions(;
    sync = true, raise = true,
    strip_llvm_debuginfo = true,
    strip = ["enzymexla.kernel_call",
             "(::Reactant.Compiler.LLVMFunc",
             "ka_with_reactant",
             "(::KernelAbstractions.Kernel",
             "var\"#_launch!;_launch!"],
)

rfirst! = if devarch isa Oceananigans.ReactantState
    @compile compile_options=compile_options first_time_step!(model)
else
    first_time_step!
end

@info "[$rank] Compiling loop..." now(UTC)

compiled_loop! = if devarch isa Oceananigans.ReactantState
    @compile compile_options=compile_options loop!(model, Ninner)
else
    loop!
end

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid)

mkpath(joinpath(profile_dir, "first_time_step"))
@info "[$rank] Running first_time_step!..." now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "first_time_step")) do
    Reactant.Profiler.annotate("bench"; metadata=Dict("step_num" => 1, "_r" => 1)) do
        @time "[$rank] first time step" rfirst!(model)
    end
end

mkpath(joinpath(profile_dir, "loop"))
@info "[$rank] running loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop")) do
    Reactant.Profiler.annotate("bench"; metadata=Dict("step_num" => 1, "_r" => 1)) do
        @time "[$rank] loop" compiled_loop!(model, Ninner)
    end
end

mkpath(joinpath(profile_dir, "loop2"))
@info "[$rank] running second loop" now(UTC)
Reactant.with_profiler(joinpath(profile_dir, "loop2")) do
    Reactant.Profiler.annotate("bench"; metadata=Dict("step_num" => 1, "_r" => 1)) do
        @time "[$rank] second loop" compiled_loop!(model, Ninner)
    end
end

checkpoint_dir = joinpath(@__DIR__, "checkpoints", replace(string(now(UTC)), ':' => '-'))
@info "[$rank] Saving checkpoint to $checkpoint_dir..." now(UTC)
GordonBell25.save_model_state(checkpoint_dir, model, arch; label="final")
@info "[$rank] Visualizing checkpoint..." now(UTC)
GordonBell25.visualize_checkpoint(joinpath(checkpoint_dir, "final");
                                  halo=8,
                                  longitude=(0, 360),
                                  latitude=(-80, 80),
                                  z=(-4000, 0))

@info "[$rank] Done!" now(UTC)
