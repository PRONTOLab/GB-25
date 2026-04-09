using Dates
using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: baroclinic_instability_model, save_model_state
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

preamble()

Ninner = ConcreteRNumber(3)
Oceananigans.defaults.FloatType = Float32

@info "Generating model..."
arch = ReactantState()
#arch = Distributed(ReactantState(), partition=Partition(2, 2, 1))
initial_conditions_path = joinpath(@__DIR__, "initial_conditions", "baroclinic_ic_quarter_degree.jld2")
model = baroclinic_instability_model(arch, resolution=8, Δt=60, Nz=10; initial_conditions_path)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true sync=true first_time_step!(model)
rstep! = @compile raise=true sync=true time_step!(model)
rloop! = @compile raise=true sync=true loop!(model, Ninner)

@info "Running..."
Reactant.with_profiler("./") do
    rfirst!(model)
end
Reactant.with_profiler("./") do
    rstep!(model)
end
Reactant.with_profiler("./") do
    rloop!(model, Ninner)
end

jobid = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH-MM-SS.sss")
checkpoint_dir = joinpath(@__DIR__, "checkpoints", jobid)
@info "Saving checkpoint..." now(UTC)
@time "checkpoint save" begin
    filepath = save_model_state(checkpoint_dir, model, arch;
        label="final", field_names=[:T, :u, :v, :w], z_indices=[:bottom, :top])
    @info "Checkpoint saved to $filepath"
end

@info "Done!"
