using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using SeawaterPolynomials
using Reactant
using Random
using Pkg

OceananigansReactantExt = Base.get_extension(Oceananigans, :OceananigansReactantExt)

Pkg.status()

include("../ocean-climate-simulation/common.jl")

raise = true

configuration = (;
    Δt                 = 1.0, #10minutes,
    resolution         = 2,
    Nz                 = 50,
    # closure            = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(),
    # closure            = Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity(),
    # closure            = Oceananigans.TurbulenceClosures.RiBasedVerticalDiffusivity(),
    # free_surface       = ExplicitFreeSurface(gravitational_acceleration=0.1),
    # buoyancy           = nothing, #BuoyancyTracer(),
    # coriolis           = nothing,
    # momentum_advection = nothing, #WENOVectorInvariant(order=5),
    # tracer_advection   = nothing, #WENO(order=5),
)

@show configuration

arch = ReactantState()
r_model = GordonBell25.baroclinic_instability_model(arch; configuration...)
c_model = GordonBell25.baroclinic_instability_model(CPU(); configuration...)
GordonBell25.sync_states!(r_model, c_model)
@show r_model isa OceananigansReactantExt.Models.ReactantHFSM 

@info "Comparing regular and Reactant model on the grid"
@show c_model.grid

@info "The Reactant model is"
@show r_model

@info "Comparing states before compilation or time stepping"
GordonBell25.compare_states(r_model, c_model)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Writing unoptimized mlir"
unopt_update_state = @code_hlo optimize=false Oceananigans.TimeSteppers.update_state!(r_model)
open("unopt_update_state.mlir", "w") do io
    show(IOContext(io, :debug => true), unopt_update_state)
end

@info "Compiling update_state..."
r_initialize! = @compile sync=true raise=raise Oceananigans.initialize!(r_model)
r_update_state! = @compile sync=true raise=raise Oceananigans.TimeSteppers.update_state!(r_model)

@time "Reactant initialize" r_initialize!(r_model)
@time "Reactant update state" r_update_state!(r_model)
@time "Regular initialize" Oceananigans.initialize!(c_model)
@time "Regular update state" Oceananigans.TimeSteppers.update_state!(c_model)

@info "After initialize and update state:"
GordonBell25.compare_states(r_model, c_model)

@info "Compiling first time step..."
r_first_time_step! = @compile sync=true raise=raise first_time_step!(r_model)

@info "Compiling time step..."
r_loop! = @compile sync=true raise=raise loop!(r_model)

@time "First Reactant time step" first_time_step!(c_model)
@time "First regular time step" r_first_time_step!(r_model)

@info "After first time step:"
GordonBell25.compare_states(r_model, c_model)

@info "Synchronizing states to test effect of initialization:"
GordonBell25.sync_states!(r_model, c_model)

@time "Ten Reactant time steps" r_loop!(r_model, ConcreteRNumber(10))
@time "Ten regular time steps" loop!(c_model, 10)

@info "After ten time steps:"
GordonBell25.compare_states(r_model, c_model)

