using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture, AbstractArchitecture, convert_to_device
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using Enzyme

using Oceananigans: initialize!, prognostic_fields, instantiated_location, boundary_conditions
using Oceananigans.Grids: AbstractGrid, XDirection, YDirection, ZDirection, inactive_cell, get_active_column_map
using Oceananigans.TimeSteppers: update_state!, ab2_step!, tick!, calculate_pressure_correction!, correct_velocities_and_cache_previous_tendencies!, step_lagrangian_particles!, ab2_step_field!, implicit_step!, pressure_correct_velocities!, cache_previous_tendencies!
using Oceananigans.Utils: @apply_regionally, launch!, configure_kernel, sum_of_velocities, KernelParameters, @constprop

using Oceananigans.Models: update_model_field_time_series!, interior_tendency_kernel_parameters, complete_communication_and_compute_buffer!

using Oceananigans.BoundaryConditions: update_boundary_condition!, 
                                       replace_horizontal_vector_halos!, 
                                       fill_halo_regions!, apply_x_bcs!,
                                       apply_y_bcs!, apply_z_bcs!, _apply_z_bcs!, apply_z_bottom_bc!, apply_z_top_bc!, 
                                       getbc, flip, update_boundary_conditions!, 
                                       fill_open_boundary_regions!, 
                                       permute_boundary_conditions, 
                                       fill_halo_event!,
                                       extract_west_bc, extract_east_bc, extract_south_bc, 
                                       extract_north_bc, extract_bc, extract_bottom_bc, extract_top_bc,
                                       fill_west_and_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!, fill_first,
                                       fill_halo_size, fill_halo_offset,
                                       _fill_bottom_and_top_halo!, _fill_bottom_halo!, _fill_top_halo!,
                                       _fill_flux_top_halo!

using Oceananigans.Fields: tupled_fill_halo_regions!, location, immersed_boundary_condition, fill_reduced_field_halos!, default_indices, ReducedField, FullField
using Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!, update_hydrostatic_pressure!
using Oceananigans.Biogeochemistry: update_biogeochemical_state!, update_tendencies!, biogeochemical_drift_velocity, biogeochemical_transition, biogeochemical_auxiliary_fields

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
                                                        _ab2_step_tracer_field!,
                                                        hydrostatic_fields

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

using Oceananigans.TurbulenceClosures: compute_diffusivities!, getclosure, clip, shear_production, dissipation, closure_turbulent_velocity, ∇_dot_qᶜ, immersed_∇_dot_qᶜ, Riᶜᶜᶠ, shear_squaredᶜᶜᶠ

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
                                           turbulent_velocityᶜᶜᶜ,
                                           convective_length_scaleᶜᶜᶠ, stability_functionᶜᶜᶠ, stable_length_scaleᶜᶜᶠ, static_column_depthᶜᶜᵃ, scale

using Oceananigans.BuoyancyFormulations: ∂z_b

using Oceananigans.Solvers: solve!, solve_batched_tridiagonal_system_kernel!
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, Az, volume, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, V⁻¹ᶜᶜᶜ, σⁿ, σ⁻, ∂zᶠᶜᶠ, δxᶠᶜᶠ, Δx⁻¹ᶠᶜᶠ
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Advection: div_Uc, _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

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
    buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState(Oceananigans.defaults.FloatType))

    # Closures:
    vertical_closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()

    tracers = (:T, :S, :e)

    grid = simple_latitude_longitude_grid(arch, Nx, Ny, Nz)

    #
    # Momentum BCs:
    #
    no_slip_bc = ValueBoundaryCondition(Field{Face, Center, Nothing}(grid))
    u_top_bc   = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    u_bcs = FieldBoundaryConditions(north=no_slip_bc, south=no_slip_bc, top=u_top_bc)

    boundary_conditions = (u=u_bcs, )

    model = HydrostaticFreeSurfaceModel(; grid,
                                          free_surface = free_surface,
                                          closure = vertical_closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          momentum_advection = nothing,
                                          tracer_advection = nothing,
                                          boundary_conditions = boundary_conditions)

    model.clock.last_Δt = Δt
    set!(model.tracers.e, 1e-6)

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

function time_step_double_gyre!(model, Tᵢ, Sᵢ, wind_stress)

    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = 1200

    # Step it forward
    Δt = model.clock.last_Δt + 0
    @trace track_numbers=false for _ = 1:2
        # Full step for tracers, fractional step for velocities.
        velocities = model.velocities
        χ = model.timestepper.χ

        Gⁿ = model.timestepper.Gⁿ.u
        G⁻ = model.timestepper.G⁻.u
        velocity_field = model.velocities.u

        launch!(model.architecture, model.grid, :xyz,
                ab2_step_field!, velocity_field, Δt, χ, Gⁿ, G⁻)

        field = velocity_field
        implicit_solver = model.timestepper.implicit_solver
        closure = model.closure
        diffusivity_fields = model.diffusivity_fields
        clock = model.clock

        solver_args = (closure, diffusivity_fields, nothing, Face(), Center(), Center(), Δt, clock)

        launch!(architecture(implicit_solver), implicit_solver.grid, :xy,
            solve_batched_tridiagonal_system_kernel!, field,
            implicit_solver.a,
            implicit_solver.b,
            implicit_solver.c,
            field,
            implicit_solver.t,
            implicit_solver.grid,
            implicit_solver.parameters,
            solver_args,
            implicit_solver.tridiagonal_direction)

        grid = model.grid

        arg_fields = fields(model)
        prog_fields = prognostic_fields(model)

        not_reduced_fields = bad_tupled_fill_halo_regions!(prog_fields, grid, model.clock, arg_fields)

        arch = model.architecture
        grid = model.grid
        velocities = model.velocities
        tracers = model.tracers
        buoyancy = model.buoyancy

        launch!(arch, grid, :xyz,
                bad_compute_CATKE_diffusivities!,
                model.diffusivity_fields, velocities)
        
        Gⁿ = model.timestepper.Gⁿ
        arch = model.architecture
        velocities = model.velocities

        launch!(arch, Gⁿ.u.grid, :xy, _bad_apply_z_bcs!, Gⁿ.u, Gⁿ.u.grid, velocities.u.boundary_conditions.top)

    end

    return nothing
end

@kernel function bad_compute_CATKE_diffusivities!(diffusivities, velocities)
    i, j, k = @index(Global, NTuple)

    Ri = 1 / (velocities.u[i,   j, k] - velocities.u[i-1, j, k])

    @inbounds diffusivities.κe[i, j, k] = 1.447 * (Ri < 0) + 0.548 * max(zero(Ri), min(one(Ri), (Ri - 0.254) / 1.02)) * (Ri ≥ 0)
end

@kernel function _bad_apply_z_bcs!(Gc, grid, top_bc)
    i, j = @index(Global, NTuple)
    @inbounds Gc[i, j, grid.Nz] -= top_bc.condition[i, j, 1]
    
end

function bad_tupled_fill_halo_regions!(fields, grid, args...; kwargs...)

    not_reduced_fields = Field[]
    for f in fields
        bcs = boundary_conditions(f)
        if !isnothing(bcs) && (!(f isa ReducedField) && f isa FullField)
            push!(not_reduced_fields, f)
        end
    end

    not_reduced_fields = tuple(not_reduced_fields...)

    c = (not_reduced_fields[1].data, )
    bcs = (not_reduced_fields[1].boundary_conditions,)
    indices = default_indices(3)
    loc = map(instantiated_location, not_reduced_fields)

    arch = architecture(grid)

    launch!(arch, grid, KernelParameters(:xy, (0, 0)),
            _bad_fill_bottom_and_top_halo!, c, grid)

    return nothing
end

@kernel function _bad_fill_bottom_and_top_halo!(c, grid)
    i, j = @index(Global, NTuple)
    @inbounds c[1][i, j, grid.Nz+1] = c[1][i, j, grid.Nz]
end


function estimate_tracer_error(model, initial_temperature, initial_salinity, wind_stress)
    time_step_double_gyre!(model, initial_temperature, initial_salinity, wind_stress)
    # Compute the mean mixed layer depth:
    Nλ, Nφ, _ = size(model.grid)
        
    return @allowscalar @inbounds model.velocities.u[1, 1, 1]
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

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, 62, 62, 15, 1200)

@info rmodel.buoyancy

rTᵢ, rSᵢ      = set_tracers(rmodel.grid)
rwind_stress = wind_stress_init(rmodel.grid)

@info "Compiling..."

dmodel = Enzyme.make_zero(rmodel)
dTᵢ = Field{Center, Center, Center}(rmodel.grid)
dSᵢ = Field{Center, Center, Center}(rmodel.grid)
dJ  = Field{Face, Center, Nothing}(rmodel.grid)

# Vanilla and forward only
vmodel = double_gyre_model(CPU(), 62, 62, 15, 1200)

vTᵢ, vSᵢ      = set_tracers(vmodel.grid)
vwind_stress = wind_stress_init(vmodel.grid)
estimate_tracer_error(vmodel, vTᵢ, vSᵢ, vwind_stress)

tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress)
println(@code_hlo raise_first=true raise=true differentiate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, dmodel, dTᵢ, dSᵢ, dJ))
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true differentiate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, dmodel, dTᵢ, dSᵢ, dJ)
compile_toc = time() - tic

@show compile_toc

# Primal-only, with reactant
pmodel = double_gyre_model(rarch, 62, 62, 15, 1200)

pTᵢ, pSᵢ      = set_tracers(pmodel.grid)
pwind_stress = wind_stress_init(pmodel.grid)

dedν, dJ = rdifferentiate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, dmodel, dTᵢ, dSᵢ, dJ)

restimate_tracer_error(pmodel, pTᵢ, pSᵢ, pwind_stress)

GordonBell25.compare_states(vmodel, pmodel; include_halos=false, throw_error=false, rtol=sqrt(eps(Float64)), atol=sqrt(eps(Float64)))
GordonBell25.compare_states(rmodel, pmodel; include_halos=true, throw_error=false, rtol=sqrt(eps(Float64)), atol=sqrt(eps(Float64)))

#=
@info "Running..."
restimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress)


@info "Running non-reactant for comparison..."
varch = CPU()
vmodel = double_gyre_model(varch, 62, 62, 15, 1200)

@info "Initialized non-reactant model"

vTᵢ, vSᵢ      = set_tracers(vmodel.grid)
vwind_stress = wind_stress_init(vmodel.grid)

@info "Initialized non-reactant tracers and wind stress"

estimate_tracer_error(vmodel, vTᵢ, vSᵢ, vwind_stress)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Done!"
=#

i = 10
j = 10

@allowscalar @show dJ[i, j]

# Produce finite-difference gradients for comparison:
ϵ_list = [1e-1, 1e-3]

@allowscalar gradient_list = Array{Float64}[]

for ϵ in ϵ_list
    rmodelP = double_gyre_model(rarch, 62, 62, 15, 1200)
    rTᵢP, rSᵢP      = set_tracers(rmodelP.grid)
    rwind_stressP = wind_stress_init(rmodelP.grid)

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, j])

    @allowscalar rwind_stressP[i, j] = rwind_stressP[i, j] + ϵ * abs(rwind_stressP[i, j])

    sq_surface_uP = restimate_tracer_error(rmodelP, rTᵢP, rSᵢP, rwind_stressP)

    rmodelM = double_gyre_model(rarch, 62, 62, 15, 1200)
    rTᵢM, rSᵢM      = set_tracers(rmodelM.grid)
    rwind_stressM = wind_stress_init(rmodelM.grid)
    @allowscalar rwind_stressM[i, j] = rwind_stressM[i, j] - ϵ * abs(rwind_stressM[i, j])

    sq_surface_uM = restimate_tracer_error(rmodelM, rTᵢM, rSᵢM, rwind_stressM)

    dsq_surface_u = (sq_surface_uP - sq_surface_uM) / diff

    #push!(gradient_list, dsq_surface_u)
    @show ϵ, dsq_surface_u

end

@info gradient_list