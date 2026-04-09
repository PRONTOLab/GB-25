using Dates
using Printf
using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant

const CHECKPOINT = joinpath(@__DIR__, "initial_conditions", "baroclinic_100day_768x768x64.jld2")

const Nx, Ny, Nz = 1536, 768, 64
const Δt         = 4minutes

@info "[serial-reactant] Building model from checkpoint" CHECKPOINT now(UTC)
arch  = ReactantState()
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz;
                                                  halo=(8, 8, 8), Δt=Δt,
                                                  initial_conditions_path = CHECKPOINT)
@info "[serial-reactant] Model built" now(UTC)

# Sanity: pull a small slice of T/S back to host to confirm interpolation populated them.
T_host = Array(interior(model.tracers.T))
S_host = Array(interior(model.tracers.S))
@info "[serial-reactant] T/S stats after init" extrema_T=extrema(T_host) extrema_S=extrema(S_host) mean_T=sum(T_host)/length(T_host) mean_S=sum(S_host)/length(S_host)

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
