using Oceananigans
using Statistics: mean

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    TKEDissipationVerticalDiffusivity,
    TKEDissipationEquations

function turbulence_simulation!(model, parameters=NamedTuple())
    N² = 1e-5
    bᵢ(z) = N² * z
    set!(model, b=bᵢ)

    @show tke_dissipation_equations = TKEDissipationEquations(; parameters...)
    closure = TKEDissipationVerticalDiffusivity(; tke_dissipation_equations)
    model.closure = closure

    @show model.closure

    for n = 1:100
        time_step!(model, model.clock.last_Δt)
    end

    return model
end

function loss_function(b_test, b_truth)
    δ = @. (b_test - b_truth)^2
    return sqrt(mean(δ))
end

function compute_loss_function(b_truth, model, parameters=NamedTuple())
    turbulence_simulation!(model, parameters)
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

    return model
end

grid = RectilinearGrid(size=32, z=(-256, 0), topology=(Flat, Flat, Bounded))
model = single_column_model(grid)
turbulence_simulation!(model)
b_truth = deepcopy(parent(model.tracers.b))

# Should be 0
@show compute_loss_function(b_truth, model)

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

@show compute_loss_function(b_truth, model, naive_parameters)
