using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: moist_baroclinic_wave_model
using GordonBell25: save_model_state
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 16,
)

default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type

using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Dates

preamble()

H = 8
Nλ = parsed_args["grid-x"] - 2H
Nφ = parsed_args["grid-y"] - 2H
Nz = parsed_args["grid-z"]

Ninner = ConcreteRNumber(2)

@info "Generating atmosphere model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz)..."
arch = ReactantState()

# Optional file-based initialization. Set the env var ATMOS_IC_PATH to a JLD2
# checkpoint produced by the spinup driver (or downloaded via
# `simulations/download_atmosphere_ic_artifact.jl`) to skip the analytic IC.
const ATMOS_IC_PATH = get(ENV, "ATMOS_IC_PATH", "")
initial_conditions_path = isempty(ATMOS_IC_PATH) ? nothing : ATMOS_IC_PATH
if initial_conditions_path !== nothing
    @info "Initializing from file" initial_conditions_path
end

model = moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz, Δt=2.0, halo=(H, H, H),
                                    initial_conditions_path)

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
    filepath = save_model_state(checkpoint_dir, model, arch;
        label="final", field_names=[:w, :T], z_indices=[:bottom, :middle, :top])
    @info "Checkpoint saved to $filepath"
end

@info "Done!"
