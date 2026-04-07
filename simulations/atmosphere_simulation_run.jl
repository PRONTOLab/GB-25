using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: moist_baroclinic_wave_model
using GordonBell25: save_model_state, visualize_checkpoint
using Oceananigans
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Dates

Oceananigans.defaults.FloatType = Float32

preamble()

H = 8
Ninner = ConcreteRNumber(2)

@info "Generating atmosphere model..."
arch = ReactantState()
model = moist_baroclinic_wave_model(arch; Nλ=48, Nφ=24, Nz=10, Δt=2.0, halo=(H, H, H))

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true sync=true first_time_step!(model)
rstep!  = @compile raise=true sync=true time_step!(model)
rloop!  = @compile raise=true sync=true loop!(model, Ninner)

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
    filepath = save_model_state(checkpoint_dir, model, arch; label="final")
    @info "Checkpoint saved to $filepath"
end

checkpoint_data_dir = joinpath(checkpoint_dir, "final")
@info "Visualizing checkpoint..." now(UTC)
@time "checkpoint visualize" begin
    visualize_checkpoint(checkpoint_data_dir; halo=H)
end

@info "Done!"
