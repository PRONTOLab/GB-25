using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init, gaussian_islands_tripolar_grid
using Oceananigans.Architectures: ReactantState
using Reactant

using ClimaOcean: ocean_simulation

using Oceananigans

using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

using ClimaOcean.Oceans: default_ocean_closure, TwoColorRadiation, default_or_override, BarotropicPotentialForcing, XDirection, YDirection

default_gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration
default_planet_rotation_rate = Oceananigans.defaults.planet_rotation_rate

using Oceananigans.BoundaryConditions: DefaultBoundaryCondition

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

function my_data_free_ocean_climate_model_init(
    arch=ReactantState();
    # Horizontal resolution
    resolution::Real = 2, # 1/4 for quarter degree
    # Vertical resolution
    Nz::Int = 20, # eventually we want to increase this to between 100-600
    )

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)

    # See visualize_ocean_climate_simulation.jl for information about how to
    # visualize the results of this run.
    Δt = 30
    free_surface = SplitExplicitFreeSurface(substeps=30)
    @allowscalar ocean = my_ocean_simulation(grid; free_surface, Δt)

    return ocean
end

function default_radiative_forcing(grid)
    ϵʳ = 0.6 # red fraction
    λʳ = 1  # red decay scale
    λᵇ = 16 # blue decay scale
    forcing = TwoColorRadiation(grid;
                                first_color_fraction = ϵʳ,
                                first_absorption_coefficient = 1/λᵇ,
                                second_absorption_coefficient = 1/λʳ)
    return forcing
end

@inline ϕ²(i, j, k, grid, ϕ)    = @inbounds ϕ[i, j, k]^2
@inline spᶠᶜᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², Φ.v))
@inline spᶜᶠᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.v[i, j, k]^2 + ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², Φ.u))

@inline u_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.u[i, j, 1] * spᶠᶜᶜ(i, j, 1, grid, Φ)
@inline v_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.v[i, j, 1] * spᶜᶠᶜ(i, j, 1, grid, Φ)

# Keep a constant linear drag parameter independent on vertical level
@inline u_immersed_bottom_drag(i, j, k, grid, clock, Φ, μ) = @inbounds - μ * Φ.u[i, j, k] * spᶠᶜᶜ(i, j, k, grid, Φ)
@inline v_immersed_bottom_drag(i, j, k, grid, clock, Φ, μ) = @inbounds - μ * Φ.v[i, j, k] * spᶜᶠᶜ(i, j, k, grid, Φ)

hasclosure(closure, ClosureType) = closure isa ClosureType
hasclosure(closure_tuple::Tuple, ClosureType) = any(hasclosure(c, ClosureType) for c in closure_tuple)

function my_ocean_simulation(grid;
                          Δt = 30,
                          closure = default_ocean_closure(),
                          tracers = (:T, :S),
                          free_surface = SplitExplicitFreeSurface(substeps=30),
                          reference_density = 1020,
                          rotation_rate = default_planet_rotation_rate,
                          gravitational_acceleration = default_gravitational_acceleration,
                          bottom_drag_coefficient = 0.003,
                          forcing = NamedTuple(),
                          biogeochemistry = nothing,
                          timestepper = :SplitRungeKutta3,
                          coriolis = HydrostaticSphericalCoriolis(; rotation_rate),
                          momentum_advection = WENOVectorInvariant(),
                          tracer_advection = WENO(order=7),
                          equation_of_state = TEOS10EquationOfState(; reference_density),
                          boundary_conditions::NamedTuple = NamedTuple(),
                          radiative_forcing = default_radiative_forcing(grid),
                          warn = true,
                          verbose = false)

    FT = eltype(grid)

    if grid isa RectilinearGrid # turn off Coriolis unless user-supplied
        coriolis = default_or_override(coriolis, nothing)
    else
        coriolis = default_or_override(coriolis)
    end

    # Detect whether we are on a single column grid
    Nx, Ny, _ = size(grid)
    single_column_simulation = Nx == 1 && Ny == 1

    if single_column_simulation
        # Let users put a bottom drag if they want
        bottom_drag_coefficient = default_or_override(bottom_drag_coefficient, zero(grid))

        # Don't let users use advection in a single column model
        tracer_advection = nothing
        momentum_advection = nothing

        # No immersed boundaries in a single column grid
        u_immersed_bc = DefaultBoundaryCondition()
        v_immersed_bc = DefaultBoundaryCondition()
    else

        bottom_drag_coefficient = default_or_override(bottom_drag_coefficient)

        u_immersed_drag = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
        v_immersed_drag = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

        u_immersed_bc = ImmersedBoundaryCondition(bottom=u_immersed_drag)
        v_immersed_bc = ImmersedBoundaryCondition(bottom=v_immersed_drag)

        # Forcing for u, v
        barotropic_potential = Field{Center, Center, Nothing}(grid)
        u_forcing = BarotropicPotentialForcing(XDirection(), barotropic_potential)
        v_forcing = BarotropicPotentialForcing(YDirection(), barotropic_potential)

        :u ∈ keys(forcing) && (u_forcing = (u_forcing, forcing[:u]))
        :v ∈ keys(forcing) && (v_forcing = (v_forcing, forcing[:v]))
        forcing = merge(forcing, (u=u_forcing, v=v_forcing))
    end

    if !isnothing(radiative_forcing)
        if :T ∈ keys(forcing)
            T_forcing = (forcing.T, radiative_forcing)
        else
            T_forcing = radiative_forcing
        end
        forcing = merge(forcing, (; T=T_forcing))
    end

    bottom_drag_coefficient = convert(FT, bottom_drag_coefficient)

    # Set up boundary conditions using Field
    top_zonal_momentum_flux      = τx = Field{Face, Center, Nothing}(grid)
    top_meridional_momentum_flux = τy = Field{Center, Face, Nothing}(grid)
    top_ocean_heat_flux          = Jᵀ = Field{Center, Center, Nothing}(grid)
    top_salt_flux                = Jˢ = Field{Center, Center, Nothing}(grid)

    # Construct ocean boundary conditions including surface forcing and bottom drag
    u_top_bc = FluxBoundaryCondition(τx)
    v_top_bc = FluxBoundaryCondition(τy)
    T_top_bc = FluxBoundaryCondition(Jᵀ)
    S_top_bc = FluxBoundaryCondition(Jˢ)

    u_bot_bc = FluxBoundaryCondition(u_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
    v_bot_bc = FluxBoundaryCondition(v_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

    default_boundary_conditions = (u = FieldBoundaryConditions(top=u_top_bc, bottom=u_bot_bc, immersed=u_immersed_bc),
                                   v = FieldBoundaryConditions(top=v_top_bc, bottom=v_bot_bc, immersed=v_immersed_bc),
                                   T = FieldBoundaryConditions(top=T_top_bc),
                                   S = FieldBoundaryConditions(top=S_top_bc))

    # Merge boundary conditions with preference to user
    # TODO: support users specifying only _part_ of the bcs for u, v, T, S (ie adding the top and immersed
    # conditions even when a user-bc is supplied).
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    buoyancy = SeawaterBuoyancy(; gravitational_acceleration, equation_of_state)

    if tracer_advection isa NamedTuple
        tracer_advection = with_tracers(tracers, tracer_advection, default_tracer_advection())
    else
        tracer_advection = NamedTuple(name => tracer_advection for name in tracers)
    end

    if hasclosure(closure, CATKEVerticalDiffusivity)
        # Turn off CATKE tracer advection
        tke_advection = (; e=nothing)
        tracer_advection = merge(tracer_advection, tke_advection)
    end

    ocean_model = HydrostaticFreeSurfaceModel(grid;
                                              buoyancy,
                                              closure,
                                              biogeochemistry,
                                              tracer_advection,
                                              momentum_advection,
                                              tracers,
                                              timestepper,
                                              free_surface,
                                              coriolis,
                                              forcing,
                                              boundary_conditions)

    ocean = Simulation(ocean_model; Δt, verbose)

    return ocean
end

@info "Generating model..."
model = my_data_free_ocean_climate_model_init(ReactantState())