using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture, AbstractArchitecture, convert_to_device
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using SeawaterPolynomials

using Enzyme

throw_error = true
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

function double_gyre_model(arch, Nx, Ny, Nz, Δt)

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30)

    # TEOS10 is a 54-term polynomial that relates temperature (T) and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType), constant_salinity=0)

    tracers = (:T, :S)

    z = [-1800.0, -570.7183557141329, -171.62903681839708, -42.06370474253316, 0.0] # hardcoded for MWE

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(8, 8, 8), z,
        longitude = (0, 360), # Problem is here: when longitude is not periodic we get error
        latitude = (15, 75),
        topology = (Periodic, Bounded, Bounded)
    )

    tracer_advection   = Centered(order=2)

    #
    # Momentum BCs:
    #
    no_slip_bc = ValueBoundaryCondition(Field{Face, Center, Nothing}(grid))
    u_top_bc   = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    u_bcs = FieldBoundaryConditions(north=no_slip_bc, south=no_slip_bc, top=u_top_bc)

    boundary_conditions = (u=u_bcs, )

    model = HydrostaticFreeSurfaceModel(; grid,
                                          free_surface = free_surface,
                                          closure = nothing, #vertical_closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          coriolis = nothing, #coriolis,
                                          momentum_advection = nothing, #momentum_advection,
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

    @inline τx(λ, φ) = cos(2π * (φ - φ₀) / Lφ)

    set!(wind_stress, τx)
    return wind_stress
end

using Oceananigans: initialize!, prognostic_fields, instantiated_location
using Oceananigans.Grids: AbstractGrid, XDirection, YDirection, ZDirection, inactive_cell, get_active_column_map
using Oceananigans.TimeSteppers: update_state!, ab2_step!, tick!, calculate_pressure_correction!, correct_velocities_and_cache_previous_tendencies!, step_lagrangian_particles!, ab2_step_field!, implicit_step!, pressure_correct_velocities!, cache_previous_tendencies!
using Oceananigans.Utils: @apply_regionally, launch!, configure_kernel

using Oceananigans.Models: update_model_field_time_series!, interior_tendency_kernel_parameters, complete_communication_and_compute_buffer!

using Oceananigans.BoundaryConditions: update_boundary_condition!, replace_horizontal_vector_halos!, fill_halo_regions!, apply_x_bcs!, apply_y_bcs!, apply_z_bcs!, _apply_z_bcs!
using Oceananigans.Fields: tupled_fill_halo_regions!, location
using Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!, update_hydrostatic_pressure!
using Oceananigans.Biogeochemistry: update_biogeochemical_state!, update_tendencies!

using Oceananigans.Models.HydrostaticFreeSurfaceModels: mask_immersed_model_fields!,
                                                        compute_tendencies!,
                                                        update_grid!,
                                                        unscale_tracers!,
                                                        compute_w_from_continuity!,
                                                        w_kernel_parameters,
                                                        p_kernel_parameters,
                                                        step_free_surface!,
                                                        compute_free_surface_tendency!,
                                                        local_ab2_step!,
                                                        ab2_step_velocities!,
                                                        ab2_step_tracers!,
                                                        compute_hydrostatic_boundary_tendency_contributions!,
                                                        compute_hydrostatic_free_surface_tendency_contributions!,
                                                        apply_flux_bcs!,
                                                        compute_hydrostatic_momentum_tendencies!,
                                                        compute_hydrostatic_free_surface_Gc!,
                                                        hydrostatic_free_surface_tracer_tendency,
                                                        barotropic_split_explicit_corrector!,
                                                        _ab2_step_tracer_field!

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: _compute_barotropic_mode!,
                                                                                  _barotropic_split_explicit_corrector!,
                                                                                  calculate_substeps,
                                                                                  calculate_adaptive_settings,
                                                                                  iterate_split_explicit!,
                                                                                  _update_split_explicit_state!,
                                                                                  _split_explicit_free_surface!,
                                                                                  _split_explicit_barotropic_velocity!,
                                                                                  compute_split_explicit_forcing!,
                                                                                  initialize_free_surface_state!,
                                                                                  initialize_free_surface_timestepper!,
                                                                                  _compute_integrated_ab2_tendencies!

using Oceananigans.TurbulenceClosures: compute_diffusivities!, getclosure, clip, shear_production, dissipation

using Oceananigans.ImmersedBoundaries: get_active_cells_map, mask_immersed_field!

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: get_top_tracer_bcs,
                                                                     update_previous_compute_time!,
                                                                     time_step_catke_equation!,
                                                                     compute_average_surface_buoyancy_flux!,
                                                                     compute_CATKE_diffusivities!,
                                                                     substep_turbulent_kinetic_energy!,
                                                                     get_time_step,
                                                                     κuᶜᶜᶠ, κcᶜᶜᶠ, κeᶜᶜᶠ,
                                                                     mask_diffusivity,
                                                                     explicit_buoyancy_flux,
                                                                     dissipation_rate,
                                                                     TKE_mixing_lengthᶜᶜᶠ,
                                                                     turbulent_velocityᶜᶜᶜ


using Oceananigans.Solvers: solve!, solve_batched_tridiagonal_system_kernel!
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll


function time_step_double_gyre!(model, wind_stress)

    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = 1200

    # Step it forward
    loop!(model)

    return nothing
end

function loop!(model)
    Δt = model.clock.last_Δt
    @trace track_numbers=false for _ = 1:11
        grid = model.grid
        Guⁿ  = model.timestepper.Gⁿ.u
        Gvⁿ  = model.timestepper.Gⁿ.v
        GUⁿ  = model.timestepper.Gⁿ.U
        GVⁿ  = model.timestepper.Gⁿ.V

        barotropic_timestepper = model.free_surface.timestepper
        baroclinic_timestepper = model.timestepper

        stage = model.clock.stage

        launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies!, GUⁿ, GVⁿ, grid,
                baroclinic_timestepper.G⁻.u, baroclinic_timestepper.G⁻.v, Guⁿ, Gvⁿ, baroclinic_timestepper.χ)

        fill!(model.free_surface.filtered_state.U, 0)

        Gⁿ = model.timestepper.Gⁿ.u
        G⁻ = model.timestepper.G⁻.u
        velocity_field = model.velocities.u

        launch!(model.architecture, model.grid, :xyz,
                ab2_step_field!, velocity_field, Δt, model.timestepper.χ, Gⁿ, G⁻)

        Gⁿ = model.timestepper.Gⁿ.T
        G⁻ = model.timestepper.G⁻.T
        tracer_field = model.tracers.T
        closure = model.closure
        grid = model.grid

        launch!(architecture(grid), grid, :xyz, _ab2_step_tracer_field!, tracer_field, grid, Δt, model.timestepper.χ, Gⁿ, G⁻)

        free_surface = model.free_surface
        baroclinic_timestepper = model.timestepper

        free_surface_grid = free_surface.η.grid
        filtered_state    = free_surface.filtered_state
        substepping       = free_surface.substepping

        arch = architecture(free_surface_grid)

        barotropic_velocities = free_surface.barotropic_velocities

        # All hardcoded
        Nsubsteps = 21
        Δτᴮ = 80.0
        weights = (-0.0027616325569592756, -0.003862908196820046, -0.0033054695005112497, -0.0010957042827565292, 0.0027497791381626267, 0.008196486562738532, 0.015182182424663676, 0.023604844197351794, 0.033308426692553815, 0.044066436251068465, 0.05556331482554762, 0.06737363395539638, 0.07893909863376815, 0.08954336106665389, 0.09828464432406694, 0.1040461758833218, 0.10546443106440806, 0.10089518635745921, 0.08837738264231575, 0.06559479830018361, 0.029835532217386728)

        # Slow forcing terms
        GUⁿ = model.timestepper.Gⁿ.U
        GVⁿ = model.timestepper.Gⁿ.V

        #free surface state
        η = free_surface.η
        U = barotropic_velocities.U
        V = barotropic_velocities.V
        η̅ = filtered_state.η
        U̅ = filtered_state.U
        V̅ = filtered_state.V

        # unpack state quantities, parameters and forcing terms
        U, V    = free_surface.barotropic_velocities
        η̅, U̅, V̅ = free_surface.filtered_state.η, free_surface.filtered_state.U, free_surface.filtered_state.V

        free_surface_kernel!, _        = configure_kernel(arch, free_surface.η.grid, free_surface.kernel_parameters, _split_explicit_free_surface!)
        barotropic_velocity_kernel!, _ = configure_kernel(arch, free_surface.η.grid, free_surface.kernel_parameters, _split_explicit_barotropic_velocity!)

        η_args = (free_surface.η.grid, Δτᴮ, free_surface.η, U, V,
                free_surface.timestepper)

        U_args = (free_surface.η.grid, Δτᴮ, free_surface.η, U, V,
                η̅, U̅, V̅, GUⁿ, GVⁿ, free_surface.gravitational_acceleration,
                free_surface.timestepper)

        for substep in 1:Nsubsteps
                averaging_weight = weights[substep]
                free_surface_kernel!(η_args...)
                barotropic_velocity_kernel!(averaging_weight, U_args...)
        end

        launch!(architecture(free_surface_grid), free_surface_grid, :xy,
                _update_split_explicit_state!, η, U, V, free_surface_grid, η̅, U̅, V̅)

        u = model.velocities.u
        v = model.velocities.v
        free_surface = model.free_surface
        state = free_surface.filtered_state
        η     = free_surface.η
        U, V  = free_surface.barotropic_velocities
        U̅, V̅  = state.U, state.V
        arch  = architecture(grid)

        launch!(architecture(grid), grid, :xy,
                _compute_barotropic_mode!,
                U̅, V̅, grid, u, v, η)

        launch!(arch, grid, :xyz, _barotropic_split_explicit_corrector!,
                u, v, U, V, U̅, V̅, η, grid)

        compute_hydrostatic_free_surface_tendency_contributions!(model, :xyz; active_cells_map=nothing)
        launch!(model.architecture, model.timestepper.Gⁿ.u.grid, :xy, _apply_z_bcs!, model.timestepper.Gⁿ.u, instantiated_location(model.timestepper.Gⁿ.u), model.timestepper.Gⁿ.u.grid, model.velocities.u.boundary_conditions.bottom, model.velocities.u.boundary_conditions.top, (model.buoyancy,))
    end
    return nothing
end

#=
function bad_compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; active_cells_map=nothing)

    arch = model.architecture
    grid = model.grid

    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; active_cells_map)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))

        @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
        @inbounds c_advection   = model.advection[tracer_name]
        @inbounds c_forcing     = model.forcing[tracer_name]

        args = tuple(Val(tracer_index),
                     Val(tracer_name),
                     c_advection,
                     model.closure,
                     model.buoyancy,
                     model.biogeochemistry,
                     model.velocities,
                     model.free_surface,
                     model.tracers,
                     model.diffusivity_fields,
                     model.auxiliary_fields,
                     model.clock,
                     c_forcing)

        launch!(arch, grid, kernel_parameters,
                compute_hydrostatic_free_surface_Gc!,
                c_tendency,
                grid,
                args;
                active_cells_map)
    end

    return nothing
end
=#

function estimate_tracer_error(model, wind_stress)
    time_step_double_gyre!(model, wind_stress)
    # Compute the mean mixed layer depth:
    Nλ, Nφ, _ = size(model.grid)
    
    mean_sq_surface_u = 0.0
    
    for j = 1:Nφ, i = 1:Nλ
        @allowscalar mean_sq_surface_u += @inbounds model.velocities.u[i, j, 1]^2
    end
    mean_sq_surface_u = mean_sq_surface_u / (Nλ * Nφ)
    
    return mean_sq_surface_u
end

function differentiate_tracer_error(model, J, dmodel, dJ)

    dedν = autodiff(set_runtime_activity(Enzyme.Reverse),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(J, dJ))

    return dedν, dJ
end

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, 62, 62, 4, 1200)

rTᵢ, rSᵢ      = set_tracers(rmodel.grid)
rwind_stress = wind_stress_init(rmodel.grid)

set!(rmodel.tracers.T, rTᵢ)

dmodel = Enzyme.make_zero(rmodel)
dJ  = Field{Face, Center, Nothing}(rmodel.grid)

@info "Compiling..."

tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(rmodel, rwind_stress)
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true differentiate_tracer_error(rmodel, rwind_stress, dmodel, dJ)
compile_toc = time() - tic

@show compile_toc

i = 10
j = 10

dedν, dJ = rdifferentiate_tracer_error(rmodel, rwind_stress, dmodel, dJ)

@allowscalar @show dJ[i, j]

# Produce finite-difference gradients for comparison:
ϵ_list = [1e-1, 1e-2, 1e-3] #, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

@allowscalar gradient_list = Array{Float64}[]

for ϵ in ϵ_list
    rmodelP = double_gyre_model(rarch, 62, 62, 4, 1200)
    rTᵢP, rSᵢP      = set_tracers(rmodelP.grid)
    rwind_stressP = wind_stress_init(rmodelP.grid)

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, j])

    @allowscalar rwind_stressP[i, j] = rwind_stressP[i, j] + ϵ * abs(rwind_stressP[i, j])

    set!(rmodelP.tracers.T, rTᵢP)
    sq_surface_uP = restimate_tracer_error(rmodelP, rwind_stressP)

    rmodelM = double_gyre_model(rarch, 62, 62, 4, 1200)
    rTᵢM, rSᵢM      = set_tracers(rmodelM.grid)
    rwind_stressM = wind_stress_init(rmodelM.grid)
    @allowscalar rwind_stressM[i, j] = rwind_stressM[i, j] - ϵ * abs(rwind_stressM[i, j])

    set!(rmodelM.tracers.T, rTᵢM)
    sq_surface_uM = restimate_tracer_error(rmodelM, rwind_stressM)

    dsq_surface_u = (sq_surface_uP - sq_surface_uM) / diff

    #push!(gradient_list, dsq_surface_u)
    @show ϵ, dsq_surface_u

end

@info gradient_list