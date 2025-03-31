# mkpath("xla_dumps")
# tmpname = tempname("xla_dumps")

# ENV["CUDA_VISIBLE_DEVICES"] = ""
# ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
# ENV["XLA_FLAGS"] = " --xla_dump_to=xla_dumps/$(tmpname)/"
ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

# @info ENV["XLA_FLAGS"]

# using MPI
# MPI.Init()  # Only needed if using MPI to detect the coordinator

using Oceananigans
using Reactant
using Dates

Reactant.Distributed.initialize()

ndevices = length(Reactant.devices())
nxdevices = floor(Int, sqrt(ndevices))
nydevices = ndevices ÷ nxdevices

arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition=Partition(nxdevices, nydevices, 1)
)

# arch = Oceananigans.ReactantState()
process_id = arch.local_rank

Δt = ConcreteRNumber(60.0)
resolution = 1/2 / ndevices
model = GordonBell25.baroclinic_instability_model(arch; resolution, Nz=32, Δt)

@show model

@info "[$(process_id)] compiling first time step" now(UTC)
compiled_first_time_step! = @compile Oceananigans.TimeSteppers.first_time_step!(model, model.clock.last_Δt)

@info "[$(process_id)] compiling second time step" now(UTC)
compiled_time_step! = @compile Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)

#=
@info "[$(process_id)] compiling update state" now(UTC)
compiled_update_state! = @compile Oceananigans.TimeSteppers.update_state!(model)
@info "[$(process_id)] compiling ab2 step" now(UTC)
compiled_ab2_step! = @compile Oceananigans.Models.HydrostaticFreeSurfaceModels.local_ab2_step!(model, Δt, 0.1)
=#

@info "[$(process_id)] running first time step" now(UTC)
@time "[$(process_id)] first time step" compiled_first_time_step!(model, model.clock.last_Δt)
@info "[$(process_id)] running loop" now(UTC)
@time "[$(process_id)] loop" for iter in 2:1000
    @info "[$(process_id)] iterating" iter now(UTC)
    @time "[$(process_id)] $(iter)-th timestep" compiled_time_step!(model, model.clock.last_Δt)
end

@info "Done!" now(UTC)

