using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, VerticallyImplicitTimeDiscretization
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

throw_error = false
include_halos = false
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

function set_tracers(grid;
                     dTdz::Real = 30.0 / 1800.0)
    fₜ(λ, φ, z) = 30 + dTdz * z # + dTdz * model.grid.Lz * 1e-6 * Ξ(z)
    Tᵢ = Field{Center, Center, Center}(grid)
    @allowscalar set!(Tᵢ, fₜ)

    return Tᵢ
end

function double_gyre_model(arch, Nx, Ny, Nz, Δt)

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30)

    # TEOS10 is a 54-term polynomial that relates temperature (T) and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(Oceananigans.defaults.FloatType), constant_salinity=0)

    # Closures:
    vertical_closure   = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization())

    tracers = (:T, :e)

    z = [-1800.0, -1328.231927341071, -978.7375431794401, -719.8257353665673, -528.019150589148, -385.9253377393035, -280.6596521340005, -202.67691422503887, -144.90588108343317, -102.10804710452429, -70.4026318872374, -46.914682599991906, -29.51438180155227, -16.62392192472553, -7.07443437500568, 0.0] # Hardcoded for MWE

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(8, 8, 8), z,
        longitude = (0, 360), # Problem is here: when longitude is not periodic we get error
        latitude = (15, 75),
        topology = (Periodic, Bounded, Bounded)
    )

    momentum_advection = VectorInvariant() #WENOVectorInvariant(order=5)
    tracer_advection   = Centered(order=2) #WENO(order=5)

    model = HydrostaticFreeSurfaceModel(; grid,
                                          free_surface = free_surface,
                                          closure = vertical_closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          coriolis = nothing,
                                          momentum_advection = momentum_advection,
                                          tracer_advection = tracer_advection)

    set!(model.tracers.e, 1e-6)

    model.clock.last_Δt = Δt

    return model
end

using Oceananigans: AbstractModel
using Oceananigans.TimeSteppers: update_state!, QuasiAdamsBashforth2TimeStepper, ReactantModel, ab2_step!, tick!, calculate_pressure_correction!, correct_velocities_and_cache_previous_tendencies!, step_lagrangian_particles!
using Oceananigans.Utils: @apply_regionally


function bad_time_step!(model, Δt;
                    callbacks=[], euler=false)

    if model.architecture == CPU()
        Δt == 0 && @warn "Δt == 0 may cause model blowup!"

        # Be paranoid and update state at iteration 0
        #model.clock.iteration == 0 && update_state!(model, callbacks; compute_tendencies=true)

        # Take an euler step if:
        #   * We detect that the time-step size has changed.
        #   * We detect that this is the "first" time-step, which means we
        #     need to take an euler step. Note that model.clock.last_Δt is
        #     initialized as Inf
        #   * The user has passed euler=true to time_step!
        euler = euler || (Δt != model.clock.last_Δt)
        euler && @debug "Taking a forward Euler step."

        # If euler, then set χ = -0.5
        minus_point_five = convert(eltype(model.grid), -0.5)
        ab2_timestepper = model.timestepper
        χ = ifelse(euler, minus_point_five, ab2_timestepper.χ)
        χ₀ = ab2_timestepper.χ # Save initial value
        ab2_timestepper.χ = χ

        # Full step for tracers, fractional step for velocities.
        ab2_step!(model, Δt)

        tick!(model.clock, Δt)
        model.clock.last_Δt = Δt
        model.clock.last_stage_Δt = Δt # just one stage

        calculate_pressure_correction!(model, Δt)
        @apply_regionally correct_velocities_and_cache_previous_tendencies!(model, Δt)
    elseif model.architecture == ReactantState()
        # Note: Δt cannot change
        if model.clock.last_Δt isa Reactant.TracedRNumber
            model.clock.last_Δt.mlir_data = Δt.mlir_data
        else
            model.clock.last_Δt = Δt
        end

        # If euler, then set χ = -0.5
        minus_point_five = convert(Float64, -0.5)
        ab2_timestepper = model.timestepper
        χ = ifelse(euler, minus_point_five, ab2_timestepper.χ)
        χ₀ = ab2_timestepper.χ # Save initial value
        ab2_timestepper.χ = χ

        # Full step for tracers, fractional step for velocities.
        ab2_step!(model, Δt)

        tick!(model.clock, Δt)

        if model.clock.last_Δt isa Reactant.TracedRNumber
            model.clock.last_Δt.mlir_data = Δt.mlir_data
        else
            model.clock.last_Δt = Δt
        end

        # just one stage
        if model.clock.last_stage_Δt isa Reactant.TracedRNumber
            model.clock.last_stage_Δt.mlir_data = Δt.mlir_data
        else
            model.clock.last_stage_Δt = Δt
        end

        calculate_pressure_correction!(model, Δt)
        correct_velocities_and_cache_previous_tendencies!(model, Δt)
    end

    update_state!(model, callbacks; compute_tendencies=true)
    step_lagrangian_particles!(model, Δt)

    # Return χ to initial value
    ab2_timestepper.χ = χ₀

    return nothing
end

function time_step_double_gyre!(model, Tᵢ)
    set!(model.tracers.T, Tᵢ)
    set!(model.velocities.u, 1)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = 1200

    # Step it forward
    Δt = model.clock.last_Δt
    update_state!(model)
    bad_time_step!(model, Δt)

    return nothing
end

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, 62, 62, 15, 1200)

rTᵢ = set_tracers(rmodel.grid)

@info "Compiling..."


tic = time()
rtime_step_double_gyre! = @compile raise_first=true raise=true sync=true time_step_double_gyre!(rmodel, rTᵢ)
compile_toc = time() - tic

@show compile_toc


@info "Running..."
rtime_step_double_gyre!(rmodel, rTᵢ)


@info "Running non-reactant for comparison..."
varch = CPU()
vmodel = double_gyre_model(varch, 62, 62, 15, 1200)

@info "Initialized non-reactant model"

vTᵢ = set_tracers(vmodel.grid)

@info "Initialized non-reactant tracers and wind stress"

time_step_double_gyre!(vmodel, vTᵢ)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Done!"
