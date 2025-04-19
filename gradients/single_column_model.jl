using Oceananigans
using Reactant
using Statistics: mean

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    TKEDissipationVerticalDiffusivity,
    TKEDissipationEquations

function bᵢ(z) # initial condition
    N² = 1e-5
    return N² * z
end

function turbulence_simulation!(model, b_initial, parameters=NamedTuple())
    copyto!(parent(model.tracers.b), b_initial)
    tke_dissipation_equations = TKEDissipationEquations(; parameters...)
    closure = TKEDissipationVerticalDiffusivity(; tke_dissipation_equations)
    model.closure = closure
    time_step!(model, model.clock.last_Δt)
    return model
end

function loss_function(b_test, b_truth)
    δ = @. (b_test - b_truth)^2
    return sqrt(mean(δ))
end

function compute_loss_function(b_truth, model, b_initial, parameters=NamedTuple())
    turbulence_simulation!(model, b_initial, parameters)
    b_test = parent(model.tracers.b)
    return loss_function(b_test, b_truth)
end

function single_column_model(grid, Δt=10 * 60)
    closure = TKEDissipationVerticalDiffusivity()

    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(-1e-3))
    b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-7))
    coriolis = FPlane(f=1e-4)

    model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis, 
        boundary_conditions = (u=u_bcs, b=b_bcs),
        tracers=(:b, :c, :e, :ϵ), buoyancy=BuoyancyTracer())

    model.clock.last_Δt = Δt
    @jit set!(model, b = bᵢ)

    return model
end

arch = Oceananigans.Architectures.ReactantState()
grid = RectilinearGrid(arch, size=32, z=(-256, 0), topology=(Flat, Flat, Bounded))
model = single_column_model(grid)
b_initial = deepcopy(parent(model.tracers.b))
@jit turbulence_simulation!(model, b_initial)
b_truth = deepcopy(parent(model.tracers.b))

# Should be 0
@jit L₀ = compute_loss_function(b_truth, model, b_initial)
@show L₀

# parameter shenanigans
defaults = (
    Cᵋϵ = 1.92,
    Cᴾϵ = 1.44,
    Cᵇϵ⁺ = -0.65,
    Cᵇϵ⁻ = -0.65,
    Cᵂu★ = 0.0,
    CᵂwΔ = 0.0,
    Cᵂα  = 0.11
)
# parameter shenanigans
naive_parameters = (
    Cᵋϵ = 1.0,
    Cᴾϵ = 1.0,
    Cᵇϵ⁺ = -1.0,
    Cᵇϵ⁻ = -1.0,
)

r_compute_loss_function = @compile sync=true raise=true compute_loss_function(b_truth, model, naive_parameters)
L₁ = r_compute_loss_function(b_truth, model, b_initial, naive_parameters) 
@show L₁