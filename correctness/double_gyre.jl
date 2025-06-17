using Enzyme

using Reactant

function wind_stress_init(cpu=true)
    res = ones(63, 63)
    if !cpu
        res = Reactant.to_rarray(res)
    end
    return res
end

function estimate_tracer_error(wind_stress)
    u = similar(wind_stress, 63, 63, 16)
    fill!(u, 0)

    v = similar(wind_stress, 78, 78, 31)
    fill!(v, 0)
    
    copyto!(@view(v[8:end-8, 8:end-8, 15]), wind_stress)

    @trace track_numbers=false for _ = 1:3

        vsl = @view(v[8:end-8, 8:end-8, 8:end-8])

        copyto!(vsl, Reactant.Ops.add(v[8:end-8, 8:end-8, 8:end-8], Reactant.TracedUtils.broadcast_to_size(u, size(vsl))))

        copyto!(u, v[9:end-7, 7:end-9, 8:end-8])

        copyto!(@view(u[:, :, 2]), u[:, :, 8])

        copyto!(vsl, Reactant.TracedUtils.broadcast_to_size(v[8:end-8, 8:end-8, 9], size(vsl)))
    end

    mean_sq_surface_u = Reactant.Ops.reduce(u, zero(eltype(u)), [1, 2, 3], +)
    
    return mean_sq_surface_u
end

function differentiate_tracer_error(J)
    dJ = zero(J)
    dedν = autodiff(set_strong_zero(Enzyme.Reverse),
                    estimate_tracer_error, Active,
                    Duplicated(J, dJ))
    return dedν, dJ
end

rwind_stress = wind_stress_init(false)

@info "Compiling..."

dJ  = make_zero(rwind_stress)

pre_pass_pipeline = pass_pipeline = "mark-func-memory-effects,inline{default-pipeline=canonicalize max-iterations=4},propagate-constant-bounds,sroa-wrappers{instcombine=false instsimplify=true },canonicalize,sroa-wrappers{instcombine=false instsimplify=true },libdevice-funcs-raise,canonicalize,remove-duplicate-func-def,canonicalize,cse,canonicalize,enzyme-hlo-generate-td{patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;noop_slice<16>;noop_reverse<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;replace_neg_add_with_subtract;binop_const_simplify;not_select_simplify;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;transpose_unary_transpose_ceil;transpose_unary_transpose_convert;transpose_unary_transpose_cosine;transpose_unary_transpose_exp;transpose_unary_transpose_expm1;transpose_unary_transpose_log;transpose_unary_transpose_log1p;transpose_unary_transpose_sign;transpose_unary_transpose_sine;transpose_unary_transpose_tanh;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;dus_licm(0);dus_pad;dus_concat;slice_dus_to_concat;slice_licm(0);pad_licm(0);elementwise_licm(0);concatenate_licm(0);slice_broadcast;associative_common_mul_op_reordering;slice_select_to_select_slice;pad_concat_to_concat_pad;slice_if;dus_to_i32;rotate_pad;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_multipad;concat_concat_to_dus;speculate_if_pad_to_select;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify;reshuffle_ands_compares;square_abs_simplify;divide_divide_simplify;concat_reshape_slice;full_reduce_reshape_or_transpose;concat_reshape_reduce;concat_elementwise;reduce_reduce;conj_real;select_broadcast_in_dim;if_op_lift_common_ops;involution_neg_simplify;involution_conj_simplify;involution_not_simplify;real_conj_simplify;conj_complex_simplify;split_convolution_into_reverse_convolution;scatter_multiply_simplify;unary_elementwise_scatter_simplify;gather_elementwise;chlo_inf_const_prop<16>;gamma_const_prop<16>;abs_const_prop<16>;log_const_prop<1>;log_plus_one_const_prop<1>;is_finite_const_prop;not_const_prop;neg_const_prop;sqrt_const_prop;rsqrt_const_prop;cos_const_prop;sin_const_prop;exp_const_prop;expm1_const_prop;tanh_const_prop;logistic_const_prop;conj_const_prop;ceil_const_prop;cbrt_const_prop;real_const_prop;imag_const_prop;round_const_prop;round_nearest_even_const_prop;sign_const_prop;floor_const_prop;tan_const_prop;add_const_prop;and_const_prop;atan2_const_prop;complex_const_prop;div_const_prop;max_const_prop;min_const_prop;mul_const_prop;or_const_prop;pow_const_prop;rem_const_prop;sub_const_prop;xor_const_prop;const_prop_through_barrier<16>;concat_const_prop<1>(1024);dynamic_update_slice_const_prop(1024);scatter_update_computation_const_prop;gather_const_prop;dus_slice_simplify;reshape_concat;reshape_dus;dot_reshape_pad<1>;pad_dot_general<1>(0);pad_dot_general<1>(1);reshape_pad;reshape_wrap;reshape_rotate;reshape_extend;reshape_slice(1);reshape_elementwise(1);transpose_select;transpose_while;transpose_slice;transpose_concat;transpose_iota;transpose_reduce;transpose_reduce_window;transpose_dus;transpose_pad<1>;transpose_einsum<1>;transpose_wrap;transpose_extend;transpose_rotate;transpose_dynamic_slice;transpose_reverse;transpose_batch_norm_training;transpose_batch_norm_inference;transpose_batch_norm_grad;transpose_if;transpose_elementwise(1);no_nan_add_sub_simplify(0);lower_extend;lower_wrap;lower_rotate},transform-interpreter,enzyme-hlo-remove-transform,enzyme" 

pass_pipeline = pre_pass_pipeline*"{postpasses=\"arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize\"},symbol-dce"

tic = time()
restimate_tracer_error = @compile optimize=pass_pipeline raise_first=true raise=true sync=true estimate_tracer_error(rwind_stress)
println(@code_hlo optimize=pass_pipeline raise_first=true raise=true estimate_tracer_error(rwind_stress))
rdifferentiate_tracer_error = @compile optimize=pass_pipeline raise_first=true raise=true sync=true differentiate_tracer_error(rwind_stress)
println(@code_hlo optimize=pre_pass_pipeline raise_first=true raise=true differentiate_tracer_error(rwind_stress))
println(@code_hlo optimize=pass_pipeline raise_first=true raise=true differentiate_tracer_error(rwind_stress))
compile_toc = time() - tic

@show compile_toc


@info "Running..."
@show restimate_tracer_error(rwind_stress)

@info "Running non-reactant for comparison..."

@info "Initialized non-reactant model"

vwind_stress = wind_stress_init(true)

@info "Initialized non-reactant tracers and wind stress"

i = 10
j = 10

dedν, dJ = rdifferentiate_tracer_error(rwind_stress)

@allowscalar @show dJ[i, j]

# Produce finite-difference gradients for comparison:
ϵ_list = [1e-1, 1e-2, 1e-3] #, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

for ϵ in ϵ_list
    rwind_stressP = wind_stress_init(false)

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, j])

    @allowscalar rwind_stressP[i, j] = rwind_stressP[i, j] + ϵ * abs(rwind_stressP[i, j])

    sq_surface_uP = restimate_tracer_error(rwind_stressP)

    rwind_stressM = wind_stress_init(false)
    @allowscalar rwind_stressM[i, j] = rwind_stressM[i, j] - ϵ * abs(rwind_stressM[i, j])

    sq_surface_uM = restimate_tracer_error(rwind_stressM)

    dsq_surface_u = (sq_surface_uP - sq_surface_uM) / diff

    @show ϵ, dsq_surface_u

end
