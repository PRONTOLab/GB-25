using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture, AbstractArchitecture, convert_to_device
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using InteractiveUtils

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
                                           TKE_mixing_lengthᶜᶜᶠ, momentum_mixing_lengthᶜᶜᶠ,
                                           turbulent_velocityᶜᶜᶜ,
                                           convective_length_scaleᶜᶜᶠ, stability_functionᶜᶜᶠ, stable_length_scaleᶜᶜᶠ, static_column_depthᶜᶜᵃ, scale

using Oceananigans.BuoyancyFormulations: ∂z_b

using Oceananigans.Solvers: solve!, solve_batched_tridiagonal_system_kernel!, get_coefficient
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, Az, volume, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, V⁻¹ᶜᶜᶜ, σⁿ, σ⁻, ∂zᶠᶜᶠ, δxᶠᶜᶠ, Δx⁻¹ᶠᶜᶠ, ℑzᵃᵃᶠ, δzᵃᵃᶠ, Δz⁻¹ᶠᶜᶠ
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.Advection: div_Uc, _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll


throw_error = true
include_halos = true
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

function resolution_to_points(resolution)
    Nx = convert(Int, 384 / resolution)
    Ny = convert(Int, 192 / resolution)
    return Nx, Ny
end

function simple_latitude_longitude_grid(arch, resolution, Nz; precompute_metrics=true)
    Nx, Ny = resolution_to_points(resolution)
    return simple_latitude_longitude_grid(arch, Nx, Ny, Nz; precompute_metrics)
end

function simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo=(8, 8, 8), precompute_metrics=true)
    z = exponential_z_faces(; Nz, depth=1800) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        longitude = (0, 360), # Problem is here: when longitude is not periodic we get error
        latitude = (15, 75),
        topology = (Periodic, Bounded, Bounded), precompute_metrics
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


@kernel function bad_solve_batched_tridiagonal_system_kernel!(ϕ, Nz, κu)
    i, j = @index(Global, NTuple)

        for k = Nz-1:-1:1
            cᵏ = @inbounds κu[i+8, j+8, k+8]

            @inbounds ϕ[i+8, j+8, k+8] -= cᵏ * (@inbounds ϕ[i+8, j+8, k+1+8] + 0.119 * 1e-3)
        end
end

function estimate_tracer_error((arch, Nx, Ny, Nz), wdata)
    vdata = similar(wdata, Nx + 8 * 2, Ny+ 8 * 2, Nz+ 8 * 2)
    fill!(vdata, 0.0)

    κu = similar(wdata, Nx + 8 * 2, Ny+ 8 * 2, Nz+ 8 * 2)
    fill!(κu, 1.0)

    edata = zero(κu)
    edata[8:end-8, 8:end-8, 9:size(vdata, 3)-7] .= 1e-3

    dev  = Oceananigans.Architectures.device(arch)

    Ri = eltype(wdata)(0.119)

    κu[8:(8+Nx-1), 9:(9+Ny-1), 10:(10+Nz-1)] .= Ri .* edata[9:(9+Nx-1), 9:(9+Ny-1), 9:(9+Nz-1)]

    vdata[:, :, Nz + 8] .= wdata[:, :, 1]

    vdata .+= 0.119 * 1e-3

    iter = vdata[9:(9+Nx), 9:(9+Ny), Nz+8][1:10, 9:10]
    
    for k = Nz-1:-1:1
        cᵏ = κu[9:(9+Nx), 9:(9+Ny), k+8][1:10, 9:10]
        iter = vdata[9:(9+Nx), 9:(9+Ny), k+8][1:10, 9:10] .- cᵏ .* iter
    end

    mean_sq_surface_u = sum(iter)

    return mean_sq_surface_u * 1e50
end

function differentiate_tracer_error(grid, wdata, dwdata)

    dedν = autodiff(set_strong_zero(Enzyme.Reverse),
                    estimate_tracer_error, Active,
                    Const(grid),
                    Duplicated(wdata, dwdata))

    return nothing
end

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, 62, 62, 15, 1200)

rvdata = parent(rmodel.velocities.u.data)

@info rmodel.buoyancy

rwind_stress = wind_stress_init(rmodel.grid)
rwdata = parent(rwind_stress.data)

@info "Compiling..."

dmodel = Enzyme.make_zero(rmodel)
dJ  = Field{Face, Center, Nothing}(rmodel.grid)
dvdata = parent(dmodel.velocities.u.data)
dwdata = parent(dJ.data)

# Vanilla and forward only
vmodel = double_gyre_model(CPU(), 62, 62, 15, 1200)

vvdata = parent(vmodel.velocities.u.data)

vwind_stress = wind_stress_init(vmodel.grid)
vwdata = parent(vwind_stress.data)
estimate_tracer_error( (vmodel.grid.architecture, vmodel.grid.Nx, vmodel.grid.Ny, vmodel.grid.Nz), vwdata)

tic = time()

passes = "mark-func-memory-effects,inline{default-pipeline=canonicalize max-iterations=4},propagate-constant-bounds,sroa-wrappers{instcombine=false instsimplify=true },canonicalize,sroa-wrappers{instcombine=false instsimplify=true },libdevice-funcs-raise,canonicalize,remove-duplicate-func-def,canonicalize,cse,canonicalize,lower-kernel{backend=cpu},canonicalize,canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-affine-to-stablehlo{prefer_while_raising=false dump_failed_lockstep=false},canonicalize,arith-raise{stablehlo=true},enzyme-batch,inline{default-pipeline=canonicalize max-iterations=4},canonicalize,cse,canonicalize,enzyme-hlo-generate-td{patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;conj_real;select_broadcast_in_dim;if_op_lift_common_ops;involution_neg_simplify;involution_conj_simplify;involution_not_simplify;real_conj_simplify;conj_complex_simplify;chlo_inf_const_prop<16>;gamma_const_prop<16>;abs_const_prop<16>;log_const_prop<1>;log_plus_one_const_prop<1>;is_finite_const_prop;not_const_prop;neg_const_prop;sqrt_const_prop;rsqrt_const_prop;cos_const_prop;sin_const_prop;exp_const_prop;expm1_const_prop;tanh_const_prop;logistic_const_prop;conj_const_prop;ceil_const_prop;cbrt_const_prop;real_const_prop;imag_const_prop;round_const_prop;round_nearest_even_const_prop;sign_const_prop;floor_const_prop;tan_const_prop;add_const_prop;and_const_prop;atan2_const_prop;complex_const_prop;div_const_prop;max_const_prop;min_const_prop;mul_const_prop;or_const_prop;pow_const_prop;rem_const_prop;sub_const_prop;xor_const_prop;const_prop_through_barrier<16>;concat_const_prop<1>(1024);dynamic_update_slice_const_prop(1024);scatter_update_computation_const_prop;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_batch_norm_training;transpose_batch_norm_inference;transpose_batch_norm_grad;transpose_if;transpose_elementwise(1);no_nan_add_sub_simplify(0);lower_extend;lower_wrap;lower_rotate},transform-interpreter,enzyme-hlo-remove-transform,enzyme{postpasses=\"arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize\"}"


passesAD = "mark-func-memory-effects,inline{default-pipeline=canonicalize max-iterations=4},propagate-constant-bounds,sroa-wrappers{instcombine=false instsimplify=true },canonicalize,sroa-wrappers{instcombine=false instsimplify=true },libdevice-funcs-raise,canonicalize,remove-duplicate-func-def,canonicalize,cse,canonicalize,lower-kernel{backend=cpu},canonicalize,canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-affine-to-stablehlo{prefer_while_raising=false dump_failed_lockstep=false},canonicalize,arith-raise{stablehlo=true},enzyme-batch,inline{default-pipeline=canonicalize max-iterations=4},canonicalize,cse,canonicalize,enzyme-hlo-generate-td{patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;conj_real;select_broadcast_in_dim;if_op_lift_common_ops;involution_neg_simplify;involution_conj_simplify;involution_not_simplify;real_conj_simplify;conj_complex_simplify;chlo_inf_const_prop<16>;gamma_const_prop<16>;abs_const_prop<16>;log_const_prop<1>;log_plus_one_const_prop<1>;is_finite_const_prop;not_const_prop;neg_const_prop;sqrt_const_prop;rsqrt_const_prop;cos_const_prop;sin_const_prop;exp_const_prop;expm1_const_prop;tanh_const_prop;logistic_const_prop;conj_const_prop;ceil_const_prop;cbrt_const_prop;real_const_prop;imag_const_prop;round_const_prop;round_nearest_even_const_prop;sign_const_prop;floor_const_prop;tan_const_prop;add_const_prop;and_const_prop;atan2_const_prop;complex_const_prop;div_const_prop;max_const_prop;min_const_prop;mul_const_prop;or_const_prop;pow_const_prop;rem_const_prop;sub_const_prop;xor_const_prop;const_prop_through_barrier<16>;concat_const_prop<1>(1024);dynamic_update_slice_const_prop(1024);scatter_update_computation_const_prop;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_batch_norm_training;transpose_batch_norm_inference;transpose_batch_norm_grad;transpose_if;transpose_elementwise(1);no_nan_add_sub_simplify(0);lower_extend;lower_wrap;lower_rotate},transform-interpreter,enzyme-hlo-remove-transform"

passes2 = "mark-func-memory-effects,inline{default-pipeline=canonicalize max-iterations=4},propagate-constant-bounds,sroa-wrappers{instcombine=false instsimplify=true },canonicalize,sroa-wrappers{instcombine=false instsimplify=true },libdevice-funcs-raise,canonicalize,remove-duplicate-func-def,canonicalize,cse,canonicalize,lower-kernel{backend=cpu},canonicalize,canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory"

passes3 = "mark-func-memory-effects,inline{default-pipeline=canonicalize max-iterations=4},propagate-constant-bounds,sroa-wrappers{instcombine=false instsimplify=true },canonicalize,sroa-wrappers{instcombine=false instsimplify=true }"

restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata)
# println(@code_hlo raise_first=true raise=true optimize=false differentiate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata, dwdata))
println(@code_hlo optimize=passes raise=true differentiate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata, dwdata))
println(@code_hlo optimize=passesAD raise=true differentiate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata, dwdata))
println(@code_hlo optimize=passes2 raise=true differentiate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata, dwdata))
println(@code_hlo optimize=passes3 raise=true differentiate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata, dwdata))
rdifferentiate_tracer_error = @compile optimize=passes raise=true sync=true differentiate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz),rwdata, dwdata)
compile_toc = time() - tic

@show compile_toc

# Primal-only, with reactant
pmodel = double_gyre_model(rarch, 62, 62, 15, 1200)

pvdata = parent(pmodel.velocities.u.data)

pwind_stress = wind_stress_init(pmodel.grid)

pwdata = parent(pwind_stress.data)

# dedν, dJ = 
rdifferentiate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata, dwdata)

restimate_tracer_error((pmodel.grid.architecture, pmodel.grid.Nx, pmodel.grid.Ny, pmodel.grid.Nz), pwdata)

GordonBell25.compare_states(vmodel, pmodel; include_halos=false, throw_error=false, rtol=sqrt(eps(Float64)), atol=sqrt(eps(Float64)))
GordonBell25.compare_states(rmodel, pmodel; include_halos=true, throw_error=false, rtol=sqrt(eps(Float64)), atol=sqrt(eps(Float64)))

#=
@info "Running..."
restimate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), rwdata)


@info "Running non-reactant for comparison..."
varch = CPU()
vmodel = double_gyre_model(varch, 62, 62, 15, 1200)

vvdata = parent(vmodel.velocities.u.data)

@info "Initialized non-reactant model"

vwind_stress = wind_stress_init(vmodel.grid)
vwdata = parent(vwind_stress.data)

@info "Initialized non-reactant tracers and wind stress"

estimate_tracer_error((vmodel.grid.architecture, vmodel.grid.Nx, vmodel.grid.Ny, vmodel.grid.Nz), vwdata)

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
    rwind_stressP = wind_stress_init(rmodelP.grid)

    rκuP = parent(rmodelP.diffusivity_fields.κu.data)
    rvdataP = parent(rmodelP.velocities.u.data)

    vwdataP = parent(rwind_stressP.data)

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, j])

    @allowscalar rwind_stressP[i, j] = rwind_stressP[i, j] + ϵ * abs(rwind_stressP[i, j])

    sq_surface_uP = restimate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), vwdataP)

    rmodelM = double_gyre_model(rarch, 62, 62, 15, 1200)
    rwind_stressM = wind_stress_init(rmodelM.grid)

    rκuM = parent(rmodelM.diffusivity_fields.κu.data)
    rvdataM = parent(rmodelM.velocities.u.data)

    vwdataM = parent(rwind_stressM.data)
    @allowscalar rwind_stressM[i, j] = rwind_stressM[i, j] - ϵ * abs(rwind_stressM[i, j])

    sq_surface_uM = restimate_tracer_error((rmodel.grid.architecture, rmodel.grid.Nx, rmodel.grid.Ny, rmodel.grid.Nz), vwdataM)

    dsq_surface_u = (sq_surface_uP - sq_surface_uM) / diff

    #push!(gradient_list, dsq_surface_u)
    @show ϵ, dsq_surface_u

end

@info gradient_list