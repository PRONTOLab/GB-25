using Oceananigans
using Oceananigans.Architectures: ReactantState
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using SeawaterPolynomials

throw_error = false
include_halos = false
rtol = sqrt(eps(Float32))
atol = sqrt(eps(Float32))

@info rtol
@info atol

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

function double_gyre_model(arch, Nx, Ny, Nz, Δt)

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30)

    # TEOS10 is a 54-term polynomial that relates temperature (T) and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType))

    # Closures:
    horizontal_closure = HorizontalScalarDiffusivity(ν = 5000.0, κ = 1000.0)
    vertical_closure   = VerticalScalarDiffusivity(ν = 1e-2, κ = 1e-5) 
    #vertical_closure   = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    #vertical_closure = Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity()
    closure = (horizontal_closure, vertical_closure)

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis()

    tracers = (:T, :S, :e, :ϵ)

    grid = simple_latitude_longitude_grid(arch, Nx, Ny, Nz)

    momentum_advection = VectorInvariant() #WENOVectorInvariant(order=5)
    tracer_advection   = Centered(order=2) #WENO(order=5)

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
                                          closure = closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          coriolis = coriolis,
                                          momentum_advection = momentum_advection,
                                          tracer_advection = tracer_advection,
                                          boundary_conditions = boundary_conditions)

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
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end

function loop!(model, Ninner)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace track_numbers=false for _ = 1:(Ninner-1)
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

function first_ts(model, Tᵢ, Sᵢ, wind_stress)
    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0

    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
end

using InteractiveUtils
using Oceananigans.TimeSteppers
using Oceananigans.TimeSteppers: calculate_pressure_correction!, correct_velocities_and_cache_previous_tendencies!, update_state!, step_lagrangian_particles!

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    step_free_surface!,
    local_ab2_step!,
    compute_free_surface_tendency!

function time_step_double_gyre!(model, Tᵢ, Sᵢ, wind_stress)
    Δt = model.clock.last_Δt
    @show @which Oceananigans.TimeSteppers.time_step!(model, Δt)
    @trace track_numbers=false for _ = 1:2

        grid = model.grid
        compute_free_surface_tendency!(grid, model, model.free_surface)

        FT = Float32
        χ = model.timestepper.χ

        # Step locally velocity and tracers
        @apply_regionally local_ab2_step!(model, Δt, χ)

        step_free_surface!(model.free_surface, model, model.timestepper, Δt)
        
    end

    return nothing
end

function estimate_tracer_error(model, initial_temperature, initial_salinity, wind_stress)
    time_step_double_gyre!(model, initial_temperature, initial_salinity, wind_stress)
    # Compute the mean mixed layer depth:
    Nλ, Nφ, _ = size(model.grid)
    
    return nothing

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

Nx = 62
Ny = 62
Nz = 15
Δt = 1200

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, Nx, Ny, Nz, Δt)


@info "Compiling..."



@info "Running non-reactant for comparison..."
varch = CPU()
vmodel = double_gyre_model(varch, Nx, Ny, Nz, Δt)

@info "Initialized non-reactant model"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

rTᵢ, rSᵢ      = set_tracers(rmodel.grid)
rwind_stress = wind_stress_init(rmodel.grid)

vTᵢ, vSᵢ      = set_tracers(vmodel.grid)
vwind_stress = wind_stress_init(vmodel.grid)

@info "Initialized non-reactant tracers and wind stress"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Running..."
@jit raise_first=true raise=true sync=true first_ts(rmodel, rTᵢ, rSᵢ, rwind_stress)
first_ts(vmodel, vTᵢ, vSᵢ, vwind_stress)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Running..."
@jit raise_first=true raise=true sync=true estimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress)
estimate_tracer_error(vmodel, vTᵢ, vSᵢ, vwind_stress)

@info Oceananigans.fields(rmodel)
@info Oceananigans.fields(vmodel)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Done!"
