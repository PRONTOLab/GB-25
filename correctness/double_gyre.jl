using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, VerticallyImplicitTimeDiscretization
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

throw_error = false
include_halos = true
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: update_state!


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

function time_step_double_gyre!(model, Tᵢ)
    set!(model.tracers.T, Tᵢ)
    set!(model.velocities.u, 1)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = 1200

    # Step it forward
    Δt = model.clock.last_Δt
    #Oceananigans.TimeSteppers.first_time_step!(model, Δt)

    #initialize!(model)
    #update_state!(model)
    time_step!(model, Δt)


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
