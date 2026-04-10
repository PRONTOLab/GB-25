using Dates
using Printf
using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Breeze.AtmosphereModels: dynamics_density

const CHECKPOINT = "/teamspace/studios/this_studio/atmos/GB-25/simulations/checkpoints/atmosphere_spinup_state_quarter_degree_2026-04-08T16-57-42.jld2"

# Quarter-degree atmosphere matching the checkpoint contents.
const Nλ, Nφ, Nz = 1440, 680, 64
const H  = 30e3
const Δt = 30.0  # checkpoint last_Δt was 30 s

@info "[serial-reactant] Building atmosphere model from checkpoint" CHECKPOINT now(UTC)
arch  = ReactantState()
model = GordonBell25.moist_baroclinic_wave_model(arch;
                                                 Nλ=Nλ, Nφ=Nφ, Nz=Nz, H=H, Δt=Δt,
                                                 initial_conditions_path = CHECKPOINT)
@info "[serial-reactant] Model built" now(UTC)

# Sanity: pull a small slice of ρ back to host to confirm initialization populated it.
ρ_host = Array(interior(dynamics_density(model.dynamics)))
@info "[serial-reactant] ρ stats after init" extrema_ρ=extrema(ρ_host) mean_ρ=sum(ρ_host)/length(ρ_host)

@info "[serial-reactant] Compiling first_time_step!..." now(UTC)
rfirst! = @compile sync=true raise=true first_time_step!(model)

Ninner = ConcreteRNumber(10)
@info "[serial-reactant] Compiling loop!..." now(UTC)
rloop! = @compile sync=true raise=true loop!(model, Ninner)

@info "[serial-reactant] Running first_time_step!..." now(UTC)
@time "[serial-reactant] first time step" rfirst!(model)

@info "[serial-reactant] Running loop (10 steps)..." now(UTC)
@time "[serial-reactant] loop10" rloop!(model, Ninner)

@info "[serial-reactant] Final clock" time=model.clock.time iteration=model.clock.iteration
@info "[serial-reactant] DONE" now(UTC)
