# "Sharded" setup which just runs on one device
#julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_instability_simulation_run.jl --float-type=Float64 --target-float-type=Float32 --dimension=first

using GordonBell25
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 8,
    grid_y_default = 4,
    grid_z_default = 4,
)

default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type

using Reactant
Reactant.set_default_backend("cpu")

if !GordonBell25.is_distributed_env_present()
    using MPI
    MPI.Init()
end

throw_error = true
include_halos = true
rtol = sqrt(eps(default_float_type))
atol = 0

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

# Single-device runs should stay on the unsharded path.
Rx, Ry = GordonBell25.factors(Ndev)
rarch = Oceananigans.ReactantState()

H = 4
Nx = parsed_args["grid-x"] 
Ny = parsed_args["grid-y"] 
Nz = parsed_args["grid-z"]

# Nx = Tx - 2H
# Ny = Ty - 2H

model_kw = (
    halo = (H, H, H),
    Δt = 1e-9,
)

varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel

# if Ndev != 1
#   @assert rmodel.architecture isa Distributed
# end

# seed both models with the same inputs
ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)

# set jit compile options for reactant
compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all, multifloat=GordonBell25.multifloat_from_args(parsed_args))

function timed_phase(f, label)
    @info "$label [start]"
    t0 = time()
    result = f()
    @info "$label [done]" elapsed_s = round(time() - t0; digits=3)
    return result
end

@info "At the beginning:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

#initialize and update_state 
Oceananigans.initialize!(vmodel)
Oceananigans.TimeSteppers.update_state!(vmodel)

timed_phase("Reactant initialize!(rmodel)") do
    @jit compile_options=compile_options Oceananigans.initialize!(rmodel)
end
timed_phase("Reactant update_state!(rmodel)") do
    @jit compile_options=compile_options Oceananigans.TimeSteppers.update_state!(rmodel)
end

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
GordonBell25.sync_states!(rmodel, vmodel)

#first time step
@showtime GordonBell25.first_time_step!(vmodel)
rfirst! = timed_phase("Compile first_time_step!(rmodel)") do
    @compile compile_options=compile_options GordonBell25.first_time_step!(rmodel)
end
@showtime rfirst!(rmodel)
@info "After first time step:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

#time_step

#Compile time_step for model
rstep! = timed_phase("Compile time_step!(rmodel)") do
    @compile compile_options=compile_options GordonBell25.time_step!(rmodel)
end

@info "Warm up rmodel:"
@showtime rstep!(rmodel)
@showtime rstep!(rmodel)

@info "Warm up vmodel:"
@showtime GordonBell25.time_step!(vmodel)
@showtime GordonBell25.time_step!(vmodel)

Nt = 10
@info "Time step with Reactant(10 steps):"
for _ in 1:Nt
    @showtime rstep!(rmodel)
end

@info "Time step vanilla(10 steps):"
for _ in 1:Nt
    @showtime GordonBell25.time_step!(vmodel)
end

@info "After $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)
@jit Oceananigans.TimeSteppers.update_state!(rmodel)
@info "After syncing and updating state again:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

# run loop of 100 steps
Nt = 100
@showtime GordonBell25.loop!(vmodel, Nt)
rNt = ConcreteRNumber(Nt)
rloop! = timed_phase("Compile loop!(rmodel, rNt)") do
    @compile compile_options=compile_options GordonBell25.loop!(rmodel, rNt)
end
@showtime rloop!(rmodel, rNt)

@info "After a loop of $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
