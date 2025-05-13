using Oceananigans
using Oceananigans.Architectures: ReactantState
using ClimaOcean
using Reactant
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using SeawaterPolynomials

function set_tracers(grid;
                     dTdz::Real = 30.0 / 1800.0)
    fₜ(λ, φ, z) = 30 + dTdz * z # + dTdz * model.grid.Lz * 1e-6 * Ξ(z)
    fₛ(λ, φ, z) = 0 #35

    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)

    @allowscalar set!(Tᵢ, fₜ)
    @allowscalar set!(Sᵢ, fₛ)

    return Tᵢ, Sᵢ
end

function resolution_to_points(resolution)
    Nx = convert(Int, 384 / resolution)
    Ny = convert(Int, 192 / resolution)
    return Nx, Ny
end

function simple_latitude_longitude_grid(arch, resolution, Nz)
    Nx, Ny = resolution_to_points(resolution)
    return simple_latitude_longitude_grid(arch, Nx, Ny, Nz)
end

function simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo=(8, 8, 8))
    z = exponential_z_faces(; Nz, depth=1800) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        longitude = (0, 360), # Problem is here: when longitude is not periodic we get error
        latitude = (15, 75),
        topology = (Bounded, Bounded, Bounded)
    )

    return grid
end

function baroclinic_instability_model(arch, Nx, Ny, Nz, Δt)

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30)

    # TEOS10 is a 54-term polynomial that relates temperature (T) and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType))

    # Closures:
    horizontal_closure = HorizontalScalarDiffusivity(ν = 5000.0, κ = 1000.0)
    #vertical_closure   = VerticalScalarDiffusivity(ν = 1e-2, κ = 1e-5) 
    vertical_closure   = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    #vertical_closure = Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity()
    closure = (horizontal_closure, vertical_closure)

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis()

    tracers = (:T, :S, :e, :ϵ)

    grid = simple_latitude_longitude_grid(arch, Nx, Ny, Nz)

    momentum_advection = VectorInvariant() #WENOVectorInvariant(order=5)
    tracer_advection   = Centered(order=2) #WENO(order=5)

    # HERE: Add boundary conditions
    #
    # Momentum BCs:
    #
    no_slip_bc = ValueBoundaryCondition(Field{Face, Center, Nothing}(grid))
    u_top_bc   = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    u_bcs = FieldBoundaryConditions(north=no_slip_bc, south=no_slip_bc, top=u_top_bc)
    v_bcs = FieldBoundaryConditions(east=no_slip_bc, west=no_slip_bc)

    boundary_conditions = (u=u_bcs, )

    model = HydrostaticFreeSurfaceModel(; grid,
                                          free_surface = free_surface,
                                          closure = vertical_closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          coriolis = coriolis,
                                          momentum_advection = momentum_advection,
                                          tracer_advection = tracer_advection,
                                          boundary_conditions = boundary_conditions)

    # Temporarily removing until we set initial tracers:
    #set!(model, T=Tᵢ, S=Sᵢ)

    model.clock.last_Δt = Δt

    return model
end

function wind_stress_init(grid;
                            ρₒ::Real = 1026.0, # kg m⁻³, average density at the surface of the world ocean
                            Lφ::Real = 60, # Meridional length in degrees
                            φ₀::Real = 15.0 # Degrees north of equator for the southern edge
                            )
    wind_stress = Field{Face, Center, Nothing}(grid)

    τ₀ = 0.1 / ρₒ # N m⁻² / density of seawater
    @inline τx(λ, φ) = τ₀ * cos(2π * (φ - φ₀) / Lφ)

    set!(wind_stress, τx)
    return wind_stress
end

function first_time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

function time_step!(model)
    Δt = model.clock.last_Δt + 0
    Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end

function loop!(model, Ninner)
    Δt = model.clock.last_Δt + 0
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace track_numbers=false for _ = 1:(Ninner-1)
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

function time_step_double_gyre!(model, Tᵢ, Sᵢ, wind_stress)

    # This doesn't currently work with Reactant:
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = 1200

    # Step it forward
    loop!(model, 10)

    return nothing
end

function estimate_tracer_error(model, initial_temperature, initial_salinity, wind_stress)
    time_step_double_gyre!(model, initial_temperature, initial_salinity, wind_stress)
    # Compute the mean mixed layer depth:
    Nλ, Nφ, _ = size(model.grid)
    
    mean_sq_surface_u = 0.0
    for j = 1:Nφ, i = 1:Nλ
        @allowscalar mean_sq_surface_u += @inbounds model.velocities.u[i, j, 1]^2
    end
    mean_sq_surface_u = mean_sq_surface_u / (Nλ * Nφ)
    return mean_sq_surface_u
end

function differentiate_tracer_error(model, Tᵢ, Sᵢ, J, dmodel, dTᵢ, dSᵢ, dJ)

    dedν = autodiff(set_runtime_activity(Enzyme.Reverse),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(Sᵢ, dSᵢ),
                    Duplicated(J, dJ))

    return dedν, dJ
end

Ninner = ConcreteRNumber(3)
Oceananigans.defaults.FloatType = Float32

@info "Generating model..."
arch = ReactantState()
model = baroclinic_instability_model(arch, 62, 62, 15, 1200)

Tᵢ, Sᵢ      = set_tracers(model.grid)
wind_stress = wind_stress_init(model.grid)

@info "Compiling..."


tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(model, Tᵢ, Sᵢ, wind_stress)
compile_toc = time() - tic

@show compile_toc


@info "Running..."
restimate_tracer_error(model, Tᵢ, Sᵢ, wind_stress)


@info "Running non-reactant for comparison..."
old_arch = CPU()
old_model = baroclinic_instability_model(old_arch, 62, 62, 15, 1200)

old_Tᵢ, old_Sᵢ      = set_tracers(old_model.grid)
old_wind_stress = wind_stress_init(old_model.grid)

estimate_tracer_error(old_model, old_Tᵢ, old_Sᵢ, old_wind_stress)

@show maximum((model.velocities.u[:] - old_model.velocities.u[:]) ./ (old_model.velocities.u[:] .+ 1))
@show maximum((model.velocities.v[1:62,1:63,1:15] - old_model.velocities.v[1:62,1:63,1:15]) ./ (old_model.velocities.v[1:62,1:63,1:15] .+ 1))

@info "Done!"
