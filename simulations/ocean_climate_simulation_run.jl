using Dates
using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: ocean_climate_model_init, save_model_state, visualize_checkpoint
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

@info "Generating model..."
model = ocean_climate_model_init(ReactantState())
Ninner = ConcreteRNumber(2)

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

checkpoint_dir = joinpath(@__DIR__, "checkpoints", replace(string(now()), ':' => '-'))
@info "Saving checkpoint to $checkpoint_dir..."
arch = ReactantState()
save_model_state(checkpoint_dir, model, arch; label="final")
@info "Visualizing checkpoint..."
visualize_checkpoint(joinpath(checkpoint_dir, "final");
                     halo=8,
                     longitude=(0, 360),
                     latitude=(-80, 80),
                     z=(-4000, 0))

@info "Done!"
