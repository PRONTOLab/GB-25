using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture, convert_to_device
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using SeawaterPolynomials

throw_error = false
include_halos = true
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

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
        topology = (Periodic, Bounded, Bounded)
    )

    return grid
end

function double_gyre_model(arch, Nx, Ny, Nz, Δt)

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30)

    # TEOS10 is a 54-term polynomial that relates temperature (T) and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType))

    # Closures:
    horizontal_closure = HorizontalScalarDiffusivity(ν = 5000, κ = 1000)
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

    set!(model.tracers.e, 1e-6)

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

using Oceananigans: initialize!, prognostic_fields
using Oceananigans.TimeSteppers: update_state!, ab2_step!, tick!, calculate_pressure_correction!, correct_velocities_and_cache_previous_tendencies!, step_lagrangian_particles!
using Oceananigans.Utils: @apply_regionally

using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: mask_immersed_model_fields!, compute_tendencies!, local_ab2_step!, compute_free_surface_tendency!, step_free_surface!
using Oceananigans.BoundaryConditions: update_boundary_condition!, replace_horizontal_vector_halos!, fill_halo_regions!
using Oceananigans.Fields: tupled_fill_halo_regions!
using Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
using Oceananigans.Biogeochemistry: update_biogeochemical_state!

using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind, mask_immersed_field!

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: calculate_substeps,
                                                                                  calculate_adaptive_settings,
                                                                                  iterate_split_explicit!,
                                                                                  _update_split_explicit_state!,
                                                                                  _split_explicit_free_surface!,
                                                                                  _split_explicit_barotropic_velocity!,
                                                                                  cache_previous_velocities!,
                                                                                  η★,
                                                                                  compute_split_explicit_forcing!,
                                                                                  initialize_free_surface_state!,
                                                                                  _compute_integrated_ab2_tendencies!


using Oceananigans.Grids: column_depthᶠᶜᵃ, column_depthᶜᶠᵃ
using Oceananigans.Utils: launch!, configure_kernel
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Operators: ∂xTᶠᶜᶠ, ∂yTᶜᶠᶠ


function time_step_double_gyre!(model, Tᵢ, Sᵢ, wind_stress)

    set!(model.tracers.T, Tᵢ)
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = 1200

    # Step it forward
    Δt = model.clock.last_Δt

    compute_tendencies!(model, [])

    grid = model.grid
    free_surface      = model.free_surface
    free_surface_grid = free_surface.η.grid


    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    baroclinic_timestepper = model.timestepper

    stage = model.clock.stage

    bad_compute_split_explicit_forcing!(GUⁿ, GVⁿ, grid, Guⁿ, Gvⁿ, baroclinic_timestepper, Val(stage))

    free_surface_grid = free_surface.η.grid
    
    # All hardcoded for mwe
    weights       = (-0.0027616325569592756, -0.003862908196820046, -0.0033054695005112497, -0.0010957042827565292, 0.0027497791381626267, 0.008196486562738532, 0.015182182424663676, 0.023604844197351794, 0.033308426692553815, 0.044066436251068465, 0.05556331482554762, 0.06737363395539638, 0.07893909863376815, 0.08954336106665389, 0.09828464432406694, 0.1040461758833218, 0.10546443106440806, 0.10089518635745921, 0.08837738264231575, 0.06559479830018361, 0.029835532217386728)
    Nsubsteps     = 21
    Δτᴮ           = 80.0

    # Slow forcing terms
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    # reset free surface averages
    bad_iterate_split_explicit!(free_surface, free_surface_grid, GUⁿ, GVⁿ, Δτᴮ, weights, Val(Nsubsteps))

    return nothing
end

@inline function bad_compute_split_explicit_forcing!(GUⁿ, GVⁿ, grid, Guⁿ, Gvⁿ, timestepper, stage)

    Gu⁻ = timestepper.G⁻.u
    Gv⁻ = timestepper.G⁻.v

    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, GUⁿ, GVⁿ, grid,
            Gu⁻, Gv⁻, Guⁿ, Gvⁿ, timestepper.χ; active_cells_map=nothing)

    return nothing
end

function bad_iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = architecture(grid)

    η           = free_surface.η
    grid        = free_surface.η.grid
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    # unpack state quantities, parameters and forcing terms
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η, state.U, state.V

    barotropic_velocity_kernel!, _ = configure_kernel(arch, grid, parameters, _bad_split_explicit_barotropic_velocity!)

    U_args = (grid, Δτᴮ, η, U, V,
              η̅, U̅, V̅, GUⁿ, GVⁿ, g,
              timestepper)

    GC.@preserve U_args begin
        barotropic_velocity_kernel!(1.0, U_args...)
    end

    return nothing
end

@kernel function _bad_split_explicit_barotropic_velocity!(averaging_weight, grid, Δτ,
                                                      η, U, V,
                                                      η̅, U̅, V̅,
                                                      Gᵁ, Gⱽ, g,
                                                      timestepper)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds V[i, j, 1] = V[i, j, 1]
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
Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, 62, 62, 15, 1200)

rTᵢ, rSᵢ      = set_tracers(rmodel.grid)
rwind_stress = wind_stress_init(rmodel.grid)

@info "Compiling..."


tic = time()
rtime_step_double_gyre! = @compile raise_first=true raise=true sync=true time_step_double_gyre!(rmodel, rTᵢ, rSᵢ, rwind_stress)
compile_toc = time() - tic

@show compile_toc


@info "Running..."
rtime_step_double_gyre!(rmodel, rTᵢ, rSᵢ, rwind_stress)


@info "Running non-reactant for comparison..."
varch = CPU()
vmodel = double_gyre_model(varch, 62, 62, 15, 1200)

@info "Initialized non-reactant model"

vTᵢ, vSᵢ      = set_tracers(vmodel.grid)
vwind_stress = wind_stress_init(vmodel.grid)

@info "Initialized non-reactant tracers and wind stress"

time_step_double_gyre!(vmodel, vTᵢ, vSᵢ, vwind_stress)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Done!"
