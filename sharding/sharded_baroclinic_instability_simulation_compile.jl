using Dates
using GordonBell25
using Reactant
using Oceananigans
using Printf

# This must be called before `GordonBell25.initialize`!
GordonBell25.preamble(; rendezvous_warn=20, rendezvous_terminate=40)
@show Ndev = length(Reactant.devices())

Rx, Ry = GordonBell25.factors(Ndev)
arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition = Partition(Rx, Ry, 1)
)

@show arch
rank = arch.local_rank

H = 8
Nx = 48 * Rx
Ny = 24 * Ry
Nz = 4

@info "[$rank] Generating model..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@show model

@jit first_time_step!(model)

# first_time_step_xla    = @code_xla raise=true GordonBell25.first_time_step!(model)
# time_step_xla          = @code_xla raise=true GordonBell25.time_step!(model)
# compute_tendencies_xla = @code_xla raise=true Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_tendencies!(model, [])
# update_state_xla       = @code_xla raise=true Oceananigans.Models.HydrostaticFreeSurfaceModels.update_state!(model)

ab2_step_xla = @code_xla raise=false Oceananigans.TimeSteppers.ab2_step!(model, model.clock.last_Δt)


codes = [
    # ("first_time_step", first_time_step_xla),
    # ("time_step", time_step_xla),
    ("ab2_step", ab2_step_xla),
    # ("compute_tendencies", compute_tendencies_xla),
    # ("update_state", update_state_xla)
]

for pair in codes
    name, code = pair             
    open("sharded_$(name)_$Ndev.xla", "w") do io
        print(io, code)
    end
end
