# Get a vanilla config with minimal size working on the cpu
# julia --project=. -O0 --threads=16 correctness/vanilla_sharded_baroclinic_instability_simulation_run.jl --float-type=Float64 --target-float-type=Float32 --dimension=first

using GordonBell25
using Oceananigans
using Oceananigans.Units
using Printf

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 8,
    grid_y_default = 4,
    grid_z_default = 4,
)

default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type

H = 4
Nx = parsed_args["grid-x"]
Ny = parsed_args["grid-y"]
Nz = parsed_args["grid-z"]

model_kw = (
    halo = (H, H, H),
    Δt = 1e-9,
)

varch = CPU()
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)

@info "Initializing vmodel..."
Oceananigans.initialize!(vmodel)

@info "Updating state for vmodel..."
Oceananigans.TimeSteppers.update_state!(vmodel)

Δt = vmodel.clock.last_Δt

@info "Running first time step..."
@time Oceananigans.TimeSteppers.time_step!(vmodel, Δt)

@info "Warm up steps:"
@time Oceananigans.TimeSteppers.time_step!(vmodel, Δt)
@time Oceananigans.TimeSteppers.time_step!(vmodel, Δt)

Nt = 10
@info "Time stepping ($Nt steps):"
for i in 1:Nt
    @time Oceananigans.TimeSteppers.time_step!(vmodel, Δt)
end

Nt_loop = 100
@info "Running a loop of $Nt_loop steps:"
@time for _ in 1:Nt_loop
    Oceananigans.TimeSteppers.time_step!(vmodel, Δt)
end

@info "Simulation complete."
