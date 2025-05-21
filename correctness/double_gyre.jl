using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture, AbstractArchitecture
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

using Oceananigans: initialize!, prognostic_fields, instantiated_location
using Oceananigans.Grids: AbstractGrid, XDirection, YDirection, ZDirection, inactive_cell
using Oceananigans.TimeSteppers: update_state!, ab2_step!, tick!, calculate_pressure_correction!, correct_velocities_and_cache_previous_tendencies!, step_lagrangian_particles!, ab2_step_field!, implicit_step!
using Oceananigans.Utils: @apply_regionally, launch!

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
                                                        apply_flux_bcs!

using Oceananigans.TurbulenceClosures: compute_diffusivities!, getclosure, clip, shear_production, dissipation

using Oceananigans.ImmersedBoundaries: get_active_cells_map

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


function time_step_double_gyre!(model, Tᵢ, Sᵢ, wind_stress)

    set!(model.tracers.T, Tᵢ)
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = 1200

    # Step it forward
    Δt = model.clock.last_Δt
    callbacks = []

    diffusivities = model.diffusivity_fields
    closure = model.closure

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy

    χ = model.timestepper.χ

    diffusivity_fields = model.diffusivity_fields

    e = model.tracers.e
    Gⁿe = model.timestepper.Gⁿ.e
    G⁻e = model.timestepper.G⁻.e

    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le
    previous_velocities = diffusivity_fields.previous_velocities
    tracer_index = findfirst(k -> k == :e, keys(model.tracers))
    implicit_solver = model.timestepper.implicit_solver

    Δτ = model.clock.last_Δt

    # Compute the linear implicit component of the RHS (diffusivities, L)
    # and step forward
    launch!(arch, grid, :xyz,
            substep_turbulent_kinetic_energy!,
            κe, Le, grid, closure,
            model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
            model.tracers, model.buoyancy, diffusivity_fields,
            Δτ, χ, Gⁿe, G⁻e)

    solver_args = (closure, diffusivity_fields, Val(tracer_index), Center(), Center(), Center(), Δτ, model.clock)

    launch!(architecture(implicit_solver), implicit_solver.grid, :xy,
            solve_batched_tridiagonal_system_kernel!, e,
            implicit_solver.a,
            implicit_solver.b,
            implicit_solver.c,
            e,
            implicit_solver.t,
            implicit_solver.grid,
            implicit_solver.parameters,
            solver_args,
            implicit_solver.tridiagonal_direction)

    launch!(arch, grid, :xyz,
            bad_compute_CATKE_diffusivities1!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    args = (model.clock, fields(model), model.closure, model.buoyancy)

    launch!(model.architecture, model.timestepper.Gⁿ.u.grid, :xy, _apply_z_bcs!, model.timestepper.Gⁿ.u, instantiated_location(model.timestepper.Gⁿ.u), model.timestepper.Gⁿ.u.grid, model.velocities.u.boundary_conditions.bottom, model.velocities.u.boundary_conditions.top, args)
    launch!(model.architecture, model.timestepper.Gⁿ.e.grid, :xy, _apply_z_bcs!, model.timestepper.Gⁿ.e, instantiated_location(model.timestepper.Gⁿ.e), model.timestepper.Gⁿ.e.grid, model.tracers.e.boundary_conditions.bottom, model.tracers.e.boundary_conditions.top, args)


    Gⁿ = model.timestepper.Gⁿ.u
    G⁻ = model.timestepper.G⁻.u

    launch!(model.architecture, model.grid, :xyz,
            ab2_step_field!, model.velocities.u, Δt, χ, Gⁿ, G⁻)

    return nothing
end

function time_step_double_gyre1!(model, Tᵢ, Sᵢ, wind_stress)
    diffusivity_fields = model.diffusivity_fields
    arch = model.architecture
    grid = model.grid

    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le

    closure = model.closure

    previous_velocities = diffusivity_fields.previous_velocities
    Δτ = model.clock.last_Δt
    χ = model.timestepper.χ

    Gⁿe = model.timestepper.Gⁿ.e
    G⁻e = model.timestepper.G⁻.e

    launch!(arch, grid, :xyz,
            bad_substep_turbulent_kinetic_energy!,
            κe, Le, grid, closure,
            model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
            model.tracers, model.buoyancy, diffusivity_fields,
            Δτ, χ, Gⁿe, G⁻e)
end


@kernel function bad_substep_turbulent_kinetic_energy!(κe, Le, grid, closure,
                                                   next_velocities, previous_velocities,
                                                   tracers, buoyancy, diffusivities,
                                                   Δτ, χ, slow_Gⁿe, G⁻e)

    i, j, k = @index(Global, NTuple)

    Jᵇ = diffusivities.Jᵇ
    e = tracers.e
    closure_ij = getclosure(i, j, closure)

    # Compute TKE diffusivity.
    @inbounds κe[i, j, k] = Oceananigans.Operators.ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, tracers.e)

    # Compute fast TKE RHS
    @inbounds P = 1 / grid.z.Δᵃᵃᶜ[k]

    @inbounds begin
        e[i, j, k] += 1000 * (P - G⁻e[i, j, k])
        G⁻e[i, j, k] = P
    end
end


@kernel function bad_compute_CATKE_diffusivities1!(diffusivities, grid, closure, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)
    Jᵇ = diffusivities.Jᵇ

    # Note: we also compute the TKE diffusivity here for diagnostic purposes, even though it
    # is recomputed in time_step_turbulent_kinetic_energy.
    κu★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)
    κc★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, Jᵇ)

    @inbounds begin
        diffusivities.κu[i, j, k] = κu★
        diffusivities.κc[i, j, k] = κc★
    end
end




Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, 62, 62, 15, 1200)

rTᵢ, rSᵢ      = set_tracers(rmodel.grid)
rwind_stress = wind_stress_init(rmodel.grid)

@info "Compiling..."

optimize="""
mark-func-memory-effects{assume_no_memory_effects=false max_iterations=8},inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },propagate-constant-bounds,sroa-wrappers{attributor=true dump_postllvm=false dump_prellvm=false instcombine=false instsimplify=true sroa=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},sroa-wrappers{attributor=true dump_postllvm=false dump_prellvm=false instcombine=false instsimplify=true sroa=true},libdevice-funcs-raise{remove_freeze=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},remove-duplicate-func-def,lower-kernel{backend=cpu},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},llvm-to-memref-access,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},convert-llvm-to-cf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-lift-cf-to-scf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(canonicalize-loops),canonicalize-scf-for,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},libdevice-funcs-raise{remove_freeze=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},affine-cfg,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(canonicalize-loops),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},llvm-to-affine-access,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},delinearize-indexing,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},simplify-affine-exprs,affine-cfg,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(affine-loop-invariant-code-motion),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},sort-memory,raise-affine-to-stablehlo{dump_failed_lockstep=false enable_lockstep_for=true err_if_not_fully_raised=true prefer_while_raising=false},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},arith-raise{stablehlo=true},enzyme-hlo-generate-td{create-module=false flags= patterns=broadcast_in_dim_op_canon<16> radix=10},transform-interpreter,
lower-jit{openmp=false backend=cpu},symbol-dce
"""


tic = time()
rtime_step_double_gyre! = @compile sync=true time_step_double_gyre!(rmodel, rTᵢ, rSᵢ, rwind_stress)
rtime_step_double_gyre1! = @compile optimize=optimize sync=true time_step_double_gyre1!(rmodel, rTᵢ, rSᵢ, rwind_stress)
compile_toc = time() - tic
@show compile_toc

@info "Running non-reactant for comparison..."
varch = CPU()
vmodel = double_gyre_model(varch, 62, 62, 15, 1200)

@info "Initialized non-reactant model"

vTᵢ, vSᵢ      = set_tracers(vmodel.grid)
vwind_stress = wind_stress_init(vmodel.grid)

@info "Initialized non-reactant tracers and wind stress"



@info "Running..."
rtime_step_double_gyre!(rmodel, rTᵢ, rSᵢ, rwind_stress)
time_step_double_gyre!(vmodel, vTᵢ, vSᵢ, vwind_stress)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Running...1"
rtime_step_double_gyre1!(rmodel, rTᵢ, rSᵢ, rwind_stress)
time_step_double_gyre1!(vmodel, vTᵢ, vSᵢ, vwind_stress)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Done!"
