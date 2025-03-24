using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using Reactant

using ClimaOcean
using ClimaOcean.OceanSimulations

using ClimaOcean.OceanSimulations: estimate_maximum_Δt, default_ocean_closure, default_free_surface, Ω_Earth, g_Earth, HydrostaticSphericalCoriolis, default_momentum_advection, TEOS10EquationOfState, default_tracer_advection, default_vertical_coordinate, Default, default_or_override, u_immersed_bottom_drag, v_immersed_bottom_drag, BarotropicPotentialForcing, XDirection, YDirection, u_quadratic_bottom_drag, v_quadratic_bottom_drag, hasclosure, CATKEVerticalDiffusivity
# using Oceananigans.Models.HydrostaticFreeSurfaceModels:
using Oceananigans.Utils: @apply_regionally, apply_regionally!

using ClimaOcean.OceanSeaIceModels.InterfaceComputations: FixedIterations, ComponentInterfaces
using OrthogonalSphericalShellGrids: TripolarGrid

using CFTime
using Dates

# https://github.com/CliMA/Oceananigans.jl/blob/da9959f3e5d8ee7cf2fb42b74ecc892874ec1687/src/AbstractOperations/conditional_operations.jl#L8
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{Oceananigans.AbstractOperations.ConditionalOperation{LX, LY, LZ, O, F, G, C, M, T}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, O, F, G, C, M, T}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)
    O2 = Reactant.traced_type_inner(O, seen, mode, track_numbers, sharding, runtime)
    F2 = Reactant.traced_type_inner(F, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    C2 = Reactant.traced_type_inner(C, seen, mode, track_numbers, sharding, runtime)
    M2 = Reactant.traced_type_inner(M, seen, mode, track_numbers, sharding, runtime)
    T2 = eltype(O2)
    return Oceananigans.AbstractOperations.ConditionalOperation{LX2, LY2, LZ2, O2, F2, G2, C2, M2, T2}
end

# https://github.com/CliMA/Oceananigans.jl/blob/da9959f3e5d8ee7cf2fb42b74ecc892874ec1687/src/AbstractOperations/kernel_function_operation.jl#L3
# struct KernelFunctionOperation{LX, LY, LZ, G, T, K, D} <: AbstractOperation{LX, LY, LZ, G, T}
Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(OA::Type{Oceananigans.AbstractOperations.KernelFunctionOperation{LX, LY, LZ, G, T, K, D}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, G, T, K, D}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    K2 = Reactant.traced_type_inner(K, seen, mode, track_numbers, sharding, runtime)
    D2 = Reactant.traced_type_inner(D, seen, mode, track_numbers, sharding, runtime)
    T2 = eltype(G2)
    return Oceananigans.AbstractOperations.KernelFunctionOperation{LX2, LY2, LZ2, G2, T2, K2, D2}
end

function mtn₁(λ, φ)
    λ₁ = 70
    φ₁ = 55
    dφ = 5
    return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
end

function mtn₂(λ, φ)
    λ₁ = 70
    λ₂ = λ₁ + 180
    φ₂ = 55
    dφ = 5
    return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
end

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, resolution, Nz)
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 180 / resolution)

    # Time step. This must be decreased as resolution is decreased.
    Δt = 1minutes

    # Grid setup
    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    #underlying_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces,
    #                                        longitude=(0, 360), latitude=(-80, 80))
    zb = z_faces[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands);
                                active_cells_map=false)
end

function MyHydrostaticFreeSurfaceModel(; grid,
                                       clock = Clock(grid),
                                       momentum_advection = VectorInvariant(),
                                       tracer_advection = Centered(),
                                       buoyancy = nothing,
                                       coriolis = nothing,
                                       free_surface = default_free_surface(grid, gravitational_acceleration=g_Earth),
                                       tracers = nothing,
                                       forcing::NamedTuple = NamedTuple(),
                                       closure = nothing,
                                       timestepper = :QuasiAdamsBashforth2,
                                       boundary_conditions::NamedTuple = NamedTuple(),
                                       particles::ParticlesOrNothing = nothing,
                                       biogeochemistry::AbstractBGCOrNothing = nothing,
                                       velocities = nothing,
                                       pressure = nothing,
                                       diffusivity_fields = nothing,
                                       auxiliary_fields = NamedTuple(),
                                       vertical_coordinate = ZCoordinate())

    # Check halos and throw an error if the grid's halo is too small
    @apply_regionally validate_model_halo(grid, momentum_advection, tracer_advection, closure)

    if !(grid isa MutableGridOfSomeKind) && (vertical_coordinate isa ZStar)
        error("The grid does not support ZStar vertical coordinates. Use a `MutableVerticalDiscretization` to allow the use of ZStar (see `MutableVerticalDiscretization`).")
    end

    # Validate biogeochemistry (add biogeochemical tracers automagically)
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    biogeochemical_fields = merge(auxiliary_fields, biogeochemical_auxiliary_fields(biogeochemistry))
    tracers, auxiliary_fields = validate_biogeochemistry(tracers, biogeochemical_fields, biogeochemistry, grid, clock)

    # Reduce the advection order in directions that do not have enough grid points
    @apply_regionally momentum_advection = validate_momentum_advection(momentum_advection, grid)
    default_tracer_advection, tracer_advection = validate_tracer_advection(tracer_advection, grid)
    default_generator(name, tracer_advection) = default_tracer_advection

    # Generate tracer advection scheme for each tracer
    tracer_advection_tuple = with_tracers(tracernames(tracers), tracer_advection, default_generator, with_velocities=false)
    momentum_advection_tuple = (; momentum = momentum_advection)
    advection = merge(momentum_advection_tuple, tracer_advection_tuple)
    advection = NamedTuple(name => adapt_advection_order(scheme, grid) for (name, scheme) in pairs(advection))

    validate_buoyancy(buoyancy, tracernames(tracers))
    buoyancy = regularize_buoyancy(buoyancy)

    # Collect boundary conditions for all model prognostic fields and, if specified, some model
    # auxiliary fields. Boundary conditions are "regularized" based on the _name_ of the field:
    # boundary conditions on u, v are regularized assuming they represent momentum at appropriate
    # staggered locations. All other fields are regularized assuming they are tracers.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.
    #
    # First, we extract boundary conditions that are embedded within any _user-specified_ field tuples:
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(pressure),
                                         extract_boundary_conditions(diffusivity_fields))

    # Next, we form a list of default boundary conditions:
    prognostic_field_names = (:u, :v, :w, tracernames(tracers)..., :η, keys(auxiliary_fields)...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(Tuple(FieldBoundaryConditions()
                                                                           for name in prognostic_field_names))

    # Then we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    boundary_conditions = merge(default_boundary_conditions, embedded_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    # Finally, we ensure that closure-specific boundary conditions, such as
    # those required by CATKEVerticalDiffusivity, are enforced:
    boundary_conditions = add_closure_specific_boundary_conditions(closure,
                                                                   boundary_conditions,
                                                                   grid,
                                                                   tracernames(tracers),
                                                                   buoyancy)

    # Ensure `closure` describes all tracers
    closure = with_tracers(tracernames(tracers), closure)

    # Put CATKE first in the list of closures
    closure = validate_closure(closure)

    # Either check grid-correctness, or construct tuples of fields
    velocities         = hydrostatic_velocity_fields(velocities, grid, clock, boundary_conditions)
    tracers            = TracerFields(tracers, grid, boundary_conditions)
    pressure           = PressureField(grid)
    diffusivity_fields = build_diffusivity_fields(diffusivity_fields, grid, clock, tracernames(tracers), boundary_conditions, closure)

    @apply_regionally validate_velocity_boundary_conditions(grid, velocities)

    arch = architecture(grid)
    free_surface = validate_free_surface(arch, free_surface)
    free_surface = materialize_free_surface(free_surface, velocities, grid)

    # Instantiate timestepper if not already instantiated
    implicit_solver   = implicit_diffusion_solver(time_discretization(closure), grid)
    prognostic_fields = hydrostatic_prognostic_fields(velocities, free_surface, tracers)

    timestepper = TimeStepper(timestepper, grid, prognostic_fields;
                              implicit_solver = implicit_solver,
                              Gⁿ = hydrostatic_tendency_fields(velocities, free_surface, grid, tracernames(tracers)),
                              G⁻ = previous_hydrostatic_tendency_fields(Val(timestepper), velocities, free_surface, grid, tracernames(tracers)))

    # Regularize forcing for model tracer and velocity fields.
    model_fields = merge(prognostic_fields, auxiliary_fields)
    forcing = model_forcing(model_fields; forcing...)
    
    model = HydrostaticFreeSurfaceModel(arch, grid, clock, advection, buoyancy, coriolis,
                                        free_surface, forcing, closure, particles, biogeochemistry, velocities, tracers,
                                        pressure, diffusivity_fields, timestepper, auxiliary_fields, vertical_coordinate)

    initialization_update_state!(model; compute_tendencies=false)

    return model
end

function my_ocean_simulation(grid;
                             Δt = estimate_maximum_Δt(grid),
                             closure = default_ocean_closure(),
                             tracers = (:T, :S),
                             free_surface = default_free_surface(grid),
                             reference_density = 1020,
                             rotation_rate = Ω_Earth,
                             gravitational_acceleration = g_Earth,
                             bottom_drag_coefficient = Default(0.003),
                             forcing = NamedTuple(),
                             biogeochemistry = nothing,
                             timestepper = :QuasiAdamsBashforth2,
                             coriolis = Default(HydrostaticSphericalCoriolis(; rotation_rate)),
                             momentum_advection = default_momentum_advection(),
                             equation_of_state = TEOS10EquationOfState(; reference_density),
                             boundary_conditions::NamedTuple = NamedTuple(),
                             tracer_advection = default_tracer_advection(),
                             vertical_coordinate = default_vertical_coordinate(grid),
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
        if warn && !(grid isa ImmersedBoundaryGrid)
            msg = """Are you totally, 100% sure that you want to build a simulation on

                   $(summary(grid))

                   rather than on an ImmersedBoundaryGrid?
                   """
            @warn msg
        end

        bottom_drag_coefficient = default_or_override(bottom_drag_coefficient)
        
        u_immersed_drag = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
        v_immersed_drag = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
        
        u_immersed_bc = ImmersedBoundaryCondition(bottom = u_immersed_drag)
        v_immersed_bc = ImmersedBoundaryCondition(bottom = v_immersed_drag)

        # Forcing for u, v
        barotropic_potential = Field{Center, Center, Nothing}(grid)
        u_forcing = BarotropicPotentialForcing(XDirection(), barotropic_potential)
        v_forcing = BarotropicPotentialForcing(YDirection(), barotropic_potential)

        :u ∈ keys(forcing) && (u_forcing = (u_forcing, forcing[:u]))
        :v ∈ keys(forcing) && (v_forcing = (v_forcing, forcing[:v]))
        forcing = merge(forcing, (u=u_forcing, v=v_forcing))
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
        # Magically add :e to tracers
        if !(:e ∈ tracers)
            tracers = tuple(tracers..., :e)
        end

        # Turn off CATKE tracer advection
        tke_advection = (; e=nothing)
        tracer_advection = merge(tracer_advection, tke_advection)
    end

    ocean_model = MyHydrostaticFreeSurfaceModel(; grid,
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
                                                boundary_conditions,
                                                vertical_coordinate)
end


function data_free_ocean_climate_simulation_init(
    arch::Architectures.AbstractArchitecture=Architectures.ReactantState();
    # Horizontal resolution
    resolution::Real = 2, # 1/4 for quarter degree
    # Vertical resolution
    Nz::Int = 20, # eventually we want to increase this to between 100-600
    output::Bool = false
    )

    grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)

    # See visualize_ocean_climate_simulation.jl for information about how to
    # visualize the results of this run.
    Δt=30seconds
    ocean = my_ocean_simulation(grid)
end # data_free_ocean_climate_simulation_init
