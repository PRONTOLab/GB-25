using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using SeawaterPolynomials
using Reactant
using Random

include("../ocean-climate-simulation/common.jl")

raise = false

configuration = (;
    Δt                 = 1.0,
    resolution         = 2,
    Nz                 = 50,
    closure            = nothing, # VerticalScalarDiffusivity(κ=1e-5, ν=1e-4),
    free_surface       = ExplicitFreeSurface(gravitational_acceleration=0.1),
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection   = WENO(order=5),
)

arch = ReactantState()
r_model = GordonBell25.baroclinic_instability_model(arch; configuration...)
c_model = GordonBell25.baroclinic_instability_model(CPU(); configuration...)
GordonBell25.sync_states!(r_model, c_model)

@info "Comparing regular and Reactant model, where the regular model is"
@show c_model
GordonBell25.compare_states(r_model, c_model)

GC.gc(true); GC.gc(false); GC.gc(true)

#@info "Compiling update_state..."
r_update_state! = @compile sync=true raise=raise Oceananigans.TimeSteppers.update_state!(r_model)

@info "Compiling first time step..."
r_first_time_step! = @compile sync=true raise=raise first_time_step!(r_model)

@info "Compiling time step..."
r_step! = @compile sync=true raise=raise time_step!(r_model)

first_time_step!(c_model)
r_first_time_step!(r_model)

@info "After first time step:"
GordonBell25.compare_states(r_model, c_model)

@time r_step!(r_model)
@time time_step!(c_model)

@info "After second time step:"
GordonBell25.compare_states(r_model, c_model)

