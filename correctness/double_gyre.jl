using Oceananigans
using Oceananigans.Architectures: ReactantState
using ClimaOcean
using Reactant
using GordonBell25
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
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

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: ab2_step_G

using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ

using Oceananigans.Grids: get_active_column_map, peripheral_node
using Oceananigans.Architectures
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index

@kernel function _compute_integrated_ab2_tendencies_bad!(Gᵁ, Gⱽ, grid, Gu⁻, Gv⁻, Guⁿ, Gvⁿ, χ)
    i, j  = @index(Global, NTuple)

    locV = (Center(), Face(), Center())

    @inbounds Gⱽ[i, j, 1] = 1000000 * bad_ab2_step_G(i, j, 1, grid, locV..., Gv⁻, Gvⁿ, χ)
end

@inline function bad_ab2_step_G(i, j, k, grid, ℓx, ℓy, ℓz, G⁻, Gⁿ, χ)
    Gⁿ⁺¹ = @inbounds Gⁿ[i, j, k] 
    immersed = peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz)

    return ifelse(immersed, zero(grid), Gⁿ⁺¹)
end

function tolaunch(model)

    grid = model.grid

    Guⁿ = model.timestepper.Gⁿ.u
    Gvⁿ = model.timestepper.Gⁿ.v

    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    baroclinic_timestepper = model.timestepper

    stage = model.clock.stage

    active_cells_map = get_active_column_map(grid)

    Gu⁻ = baroclinic_timestepper.G⁻.u
    Gv⁻ = baroclinic_timestepper.G⁻.v

    launch!(architecture(grid), grid, :xy, _compute_integrated_ab2_tendencies_bad!, GUⁿ, GVⁿ, grid,
            Gu⁻, Gv⁻, Guⁿ, Gvⁿ, baroclinic_timestepper.χ; active_cells_map)
end

function time_step_double_gyre!(model, Tᵢ, Sᵢ, wind_stress)
    Δt = model.clock.last_Δt
    @show @which Oceananigans.TimeSteppers.time_step!(model, Δt)
    @trace track_numbers=false for _ = 1:2

        @apply_regionally tolaunch(model)
        
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

optimize="""
mark-func-memory-effects{assume_no_memory_effects=false max_iterations=8},inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },propagate-constant-bounds,sroa-wrappers{attributor=true dump_postllvm=false dump_prellvm=false instcombine=false instsimplify=true sroa=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},sroa-wrappers{attributor=true dump_postllvm=false dump_prellvm=false instcombine=false instsimplify=true sroa=true},libdevice-funcs-raise{remove_freeze=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},remove-duplicate-func-def,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0) radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0);lower_extend;lower_wrap;lower_rotate radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,enzyme-batch,inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0) radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0);lower_extend;lower_wrap;lower_rotate radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,
inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0) radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0);lower_extend;lower_wrap;lower_rotate radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},remove-unnecessary-enzyme-ops,enzyme-simplify-math,inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0) radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;chlo_inf_const_prop<16>;gamma_const_prop<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;abs_const_prop<16>;negate_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;sqrt_simplify<16>;cos_simplify<16>;sin_simplify<16>;noop_slice<16>;noop_reverse<16>;const_prop_through_barrier<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;tanh_simplify<16>;exp_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;gather_simplify<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_const_prop<1>(1024);concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_update_slice_const_prop(1024);dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;scatter_indices_are_unique;replace_neg_add_with_subtract;log_const_prop<1>;log_plus_one_const_prop<1>;binop_const_simplify;is_finite_const_prop;not_const_prop;not_select_simplify;scatter_update_computation_const_prop;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;dus_licm(0);while_op_induction_replacement;dus_pad;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;while_pad_induction_reduction;while_licm<1>(1);associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_elementwise(1);no_nan_add_sub_simplify(0);lower_extend;lower_wrap;lower_rotate radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,lower-kernel{backend=cpu},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},llvm-to-memref-access,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},convert-llvm-to-cf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-lift-cf-to-scf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(canonicalize-loops),canonicalize-scf-for,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},libdevice-funcs-raise{remove_freeze=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
affine-cfg,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(canonicalize-loops),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},llvm-to-affine-access,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},delinearize-indexing,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},simplify-affine-exprs,affine-cfg,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(affine-loop-invariant-code-motion),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},sort-memory,
raise-affine-to-stablehlo{dump_failed_lockstep=false enable_lockstep_for=true err_if_not_fully_raised=true prefer_while_raising=false},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},arith-raise{stablehlo=true},
inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;
transpose_transpose<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
conj_complex_negate<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
chlo_inf_const_prop<16>;
gamma_const_prop<16>;
cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_abs<16>;
cse_concatenate<16>;
concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
xor_simplify<16>;
abs_const_prop<16>;
negate_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
simplify_extend<16>;
simplify_wrap<16>;
simplify_rotate<16>;
sqrt_simplify<16>;
cos_simplify<16>;
sin_simplify<16>;
noop_slice<16>;
noop_reverse<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>(1024);
select_pad_to_dus<1>;
and_pad_pad<1>;
negative_pad_to_slice<16>;
tanh_simplify<16>;
exp_simplify<16>;
slice_simplify<16>;
convert_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
gather_simplify<16>;
slice_internal;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>(1024);
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
binop_const_reshape_pad<1>;
binop_const_pad_add<1>;
binop_const_pad_subtract<1>;
binop_const_pad_mul<1>;
binop_const_pad_div<1>;
binop_binop_pad_pad_add<1>;
binop_binop_pad_pad_mul<1>;
binop_pad_pad_add<1>;
binop_pad_pad_subtract<1>;
binop_pad_pad_mul<1>;
binop_pad_pad_div<1>;
binop_pad_pad_min<1>;
binop_pad_pad_max<1>;
unary_pad_push_convert<1>;
unary_pad_push_tanh<1>;
unary_pad_push_exp<1>;
transpose_dot_reorder<1>;
dot_transpose<1>;
transpose_convolution<1>;
convolution_transpose<1>;
convert_convert_float<1>;
concat_to_pad<1>;

transpose_unary_transpose_expm1;
transpose_unary_transpose_log;
transpose_unary_transpose_log1p;
transpose_unary_transpose_sign;
transpose_unary_transpose_sine;
transpose_unary_transpose_tanh;
select_comp_iota_const_simplify<1>;
sign_abs_simplify<1>;
broadcastindim_is_reshape;
slice_reduce_window<1>;
while_deadresult;
while_dus;

dus_licm(0);
while_op_induction_replacement;
dus_pad;
dus_concat;
slice_dus_to_concat;
while_induction_reduction;
slice_licm(0);
pad_licm(0);
elementwise_licm(0);
concatenate_licm(0);
slice_broadcast;
while_pad_induction_reduction;
while_licm<1>(1);

broadcast_iota_simplify;
select_comp_iota_to_dus;
compare_cleanup;
broadcast_compare;
not_compare;
broadcast_iota;
cse_iota;
compare_iota_const_simplify;
reshuffle_ands_compares;
square_abs_simplify;
divide_divide_simplify;
concat_reshape_slice;
full_reduce_reshape_or_transpose;
concat_reshape_reduce;
concat_elementwise;

reduce_reduce;
dus_slice_simplify;
reshape_concat;
reshape_dus;
dot_reshape_pad<1>;
pad_dot_general<1>(0);
pad_dot_general<1>(1);
reshape_pad;
reshape_wrap;
reshape_rotate;
reshape_extend;
reshape_slice(1);
reshape_elementwise(1);
no_nan_add_sub_simplify(0) radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform
"""
@info "Running..."
@jit optimize=optimize sync=true first_ts(rmodel, rTᵢ, rSᵢ, rwind_stress)
first_ts(vmodel, vTᵢ, vSᵢ, vwind_stress)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Running..."
@jit optimize=optimize sync=true estimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress)
estimate_tracer_error(vmodel, vTᵢ, vSᵢ, vwind_stress)

@info Oceananigans.fields(rmodel)
@info Oceananigans.fields(vmodel)

GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "Done!"
