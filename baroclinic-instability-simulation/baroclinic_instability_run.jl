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

Oceananigans.defaults.FloatType = Float32
raise = true

configuration = (;
    Δt                 = 1minutes, #10minutes,
    resolution         = 2,
    Nz                 = 50,
    closure            = nothing,
    # closure            = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(),
    # closure            = Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity(),
    # closure            = Oceananigans.TurbulenceClosures.RiBasedVerticalDiffusivity(),
    # free_surface       = ExplicitFreeSurface(gravitational_acceleration=0.1),
    # buoyancy           = BuoyancyTracer(),
    coriolis           = nothing,
    # momentum_advection = nothing, #WENOVectorInvariant(order=5),
    # tracer_advection   = nothing, #WENO(order=5),
)

@show configuration

#arch = ReactantState()
arch = Distributed(ReactantState(), partition=Partition(2, 2, 1))
r_model = GordonBell25.baroclinic_instability_model(arch; configuration...)
@show r_model isa OceananigansReactantExt.Models.ReactantHFSM 

@info "The Reactant model is"
@show r_model

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling update_state..."
r_initialize! = @compile sync=true raise=raise Oceananigans.initialize!(r_model)
r_update_state! = @compile sync=true raise=raise Oceananigans.TimeSteppers.update_state!(r_model)

@info "Compiling first time step..."
r_first_time_step! = @compile sync=true raise=raise first_time_step!(r_model)

@info "Compiling loop..."
r_loop! = @compile sync=true raise=raise loop!(r_model, ConcreteRNumber(10))

@time "Reactant initialize" r_initialize!(r_model)
@time "Reactant update state" r_update_state!(r_model)
@time "First Reactant time step" r_first_time_step!(r_model)
@time "First Reactant time step" r_first_time_step!(r_model)
@time "First Reactant time step" r_first_time_step!(r_model)
@time "Reactant ten step loop" r_loop!(r_model, ConcreteRNumber(10))
@time "Reactant ten step loop" r_loop!(r_model, ConcreteRNumber(10))
@time "Reactant ten step loop" r_loop!(r_model, ConcreteRNumber(10))

