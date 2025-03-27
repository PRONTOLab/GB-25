using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using SeawaterPolynomials
using Reactant
using Random

include("../ocean-climate-simulation/common.jl")

Δt = 2minutes
resolution = 2
closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
configuration = (; Δt, resolution, closure)

arch = Distributed(ReactantState(), partition=Partition(2, 1))
r_model = GordonBell25.baroclinic_instability_model(arch; configuration...)
c_model = GordonBell25.baroclinic_instability_model(CPU(); configuration...)
GordonBell25.sync_states!(r_model, c_model)

@info "Comparing regular and Reactant model, where the regular model is"
@show c_model
GordonBell25.compare_states(r_model, c_model)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling update_state..."
raise = false
r_update_state! = @compile sync=true raise=raise Oceananigans.TimeSteppers.update_state!(r_model)

@info "Compiling first time step..."
r_first_time_step! = @compile sync=true raise=raise first_time_step!(r_model)

@info "Compiling time step..."
r_step! = @compile sync=true raise=raise time_step!(r_model)

first_time_step!(c_model)

r_update_state!(r_model) # additionally needed for Reactant
r_first_time_step!(r_model)

@info "After first time step:"
GordonBell25.compare_states(r_model, c_model)

@time r_step!(r_model)
@time time_step!(c_model)

@info "After second time step:"
GordonBell25.compare_states(r_model, c_model)

