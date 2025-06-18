using Reactant

using Enzyme

function double_gyre_model()
    pv02 = Reactant.to_rarray(ones(78, 31))
    return pv02
end

function wind_stress_init()
    res = ones(63, 1)
    res = Reactant.to_rarray(res)
    return res
end


txt = """
  func.func @main(%arg1: tensor<31x78xf64>, %arg2: tensor<1x63xf64>) -> (tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c = stablehlo.constant dense<7> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<78x31xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<63x16xf64>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<63x1xf64>
    %1 = stablehlo.reshape %arg2 : (tensor<1x63xf64>) -> tensor<63x1xf64>
    %3:3 = stablehlo.while(%iterArg = %c_0, %iterArg_5 = %cst_1, %iterArg_6 = %cst_4) : tensor<i64>, tensor<63x1xf64>, tensor<63x16xf64> attributes {enzymexla.disable_min_cut}
     cond {
      %9 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %9 : tensor<i1>
    } do {
      %9 = stablehlo.add %iterArg, %c_2 : tensor<i64>

      %16 = stablehlo.broadcast_in_dim %iterArg_5, dims = [0, 1] : (tensor<63x1xf64>) -> tensor<63x16xf64>

      %10 = stablehlo.add %16, %iterArg_6 : tensor<63x16xf64>

      %11 = stablehlo.dynamic_update_slice %cst_3, %10, %c, %c : (tensor<78x31xf64>, tensor<63x16xf64>, tensor<i32>, tensor<i32>) -> tensor<78x31xf64>
      %12 = stablehlo.slice %11 [8:71, 7:8] : (tensor<78x31xf64>) -> tensor<63x1xf64>
      %13 = stablehlo.slice %11 [8:71, 9:23] : (tensor<78x31xf64>) -> tensor<63x14xf64>
      %14 = stablehlo.concatenate %12, %1, %13, dim = 1 : (tensor<63x1xf64>, tensor<63x1xf64>, tensor<63x14xf64>) -> tensor<63x16xf64>
      %15 = stablehlo.slice %10 [0:63, 1:2] : (tensor<63x16xf64>) -> tensor<63x1xf64>
      stablehlo.return %9, %15, %14 : tensor<i64>, tensor<63x1xf64>, tensor<63x16xf64>
    }
    %6 = stablehlo.reduce(%3#2 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<63x16xf64>, tensor<f64>) -> tensor<f64>
    return %6 : tensor<f64>
  }
"""
function estimate_tracer_error(pv02, wind_stress) 
    return (Reactant.Ops.hlo_call(txt, Reactant.materialize_traced_array(transpose(pv02)), Reactant.materialize_traced_array(transpose(wind_stress))))[1]

    u = similar(v, 63, 16)
    fill!(u, 0)
    
    v0 = copy(v)

    pv2 = similar(u, 78, 31)
    fill!(pv2, 0)

    @trace track_numbers=false for _ = 1:3
        copyto!(@view(v[8:end-8, 8:end-8]), Reactant.Ops.add(v[8:end-8, 8:end-8], u))

        copyto!(u, v[9:end-7, 8:end-8])

        copyto!(@view(u[:, 2:2]), wind_stress)

        sVp = Reactant.TracedUtils.broadcast_to_size(v[8:end-8, 9:9], size(v[8:end-8, 8:end-8]))

        copyto!(@view(v[8:end-8, 8:end-8]), sVp)

        copyto!(@view(pv2[8:end-8, 8:end-8]), sVp)

    end

    copyto!(pv02, pv2)

    # adding this fixes
    # copyto!(v, v0)

    return Reactant.Ops.reduce(u, eltype(u)(0), [1, 2], +)

    # mean_sq_surface_u = sum(u)
    
    return mean_sq_surface_u
end

function differentiate_tracer_error(pv02, J, dJ)
    dpv02 = zero(pv02)
    dJ = copy(dJ)

    autodiff(set_strong_zero(Enzyme.Reverse),
                    estimate_tracer_error, Active,
                    Duplicated(pv02, dpv02),
                    Duplicated(J, dJ))

    return dJ
end

rmodel = double_gyre_model()
rwind_stress = wind_stress_init()

@info "Compiling..."

dJ  = make_zero(rwind_stress) # Field{Face, Center, Nothing}(rmodel.grid)

pre_pipeline = "inline{default-pipeline=canonicalize max-iterations=4},cse,canonicalize,enzyme-hlo-generate-td{patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;gather_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_pad<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1024);select_op_canon<16>(1024);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;noop_slice<16>;noop_reverse<16>;slice_slice<16>;shift_right_logical_simplify<16>;pad_simplify<16>(1024);select_pad_to_dus<1>;and_pad_pad<1>;negative_pad_to_slice<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;slice_internal;iota_simplify<16>(1024);broadcast_in_dim_simplify<16>(1024);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;slice_pad<1>;dot_reshape_dot<1>;concat_fuse<1>;pad_reshape_pad<1>;pad_pad<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;scatter_to_dynamic_update_slice<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;add_pad_pad_to_concat<1>;broadcast_reshape<1>;concat_pad<1>;reduce_pad<1>;broadcast_pad<1>;zero_product_reshape_pad<1>;mul_zero_pad<1>;div_zero_pad<1>;binop_const_reshape_pad<1>;binop_const_pad_add<1>;binop_const_pad_subtract<1>;binop_const_pad_mul<1>;binop_const_pad_div<1>;binop_binop_pad_pad_add<1>;binop_binop_pad_pad_mul<1>;binop_pad_pad_add<1>;binop_pad_pad_subtract<1>;binop_pad_pad_mul<1>;binop_pad_pad_div<1>;binop_pad_pad_min<1>;binop_pad_pad_max<1>;unary_pad_push_convert<1>;unary_pad_push_tanh<1>;unary_pad_push_exp<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;concat_to_pad<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;dynamic_gather_op_is_not_dynamic<16>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;replace_neg_add_with_subtract;binop_const_simplify;not_select_simplify;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_unary_transpose_abs;transpose_unary_transpose_neg;transpose_unary_transpose_sqrt;transpose_unary_transpose_rsqrt;while_deadresult;while_dus},transform-interpreter,enzyme-hlo-remove-transform,canonicalize,cse,canonicalize"

pass_pipeline = pre_pipeline * ",enzyme{postpasses=\"arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize\"},symbol-dce"

tic = time()

println(@code_hlo optimize=pass_pipeline raise_first=true raise=true estimate_tracer_error(rmodel, rwind_stress))
println(@code_hlo optimize=pre_pipeline raise_first=true raise=true differentiate_tracer_error(rmodel, rwind_stress, dJ))
println(@code_hlo optimize=pass_pipeline raise_first=true raise=true differentiate_tracer_error(rmodel, rwind_stress, dJ))


restimate_tracer_error = @compile optimize=pass_pipeline raise_first=true raise=true sync=true estimate_tracer_error(rmodel, rwind_stress)
rdifferentiate_tracer_error = @compile optimize=pass_pipeline raise_first=true raise=true sync=true differentiate_tracer_error(rmodel, rwind_stress, dJ)


compile_toc = time() - tic

@show compile_toc

i = 10

dJ = rdifferentiate_tracer_error(rmodel, rwind_stress, dJ)

@allowscalar @show dJ[i, 1]

# Produce finite-difference gradients for comparison:
ϵ_list = [1e-1, 1e-2, 1e-3] #, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

@allowscalar gradient_list = Array{Float64}[]

for ϵ in ϵ_list
    rmodelP = double_gyre_model()
    rwind_stressP = wind_stress_init()

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, 1])

    @allowscalar rwind_stressP[i, 1] = rwind_stressP[i, 1] + ϵ * abs(rwind_stressP[i, 1])

    sq_surface_uP = restimate_tracer_error(rmodelP, rwind_stressP)

    rmodelM = double_gyre_model()
    rwind_stressM = wind_stress_init()
    @allowscalar rwind_stressM[i, 1] = rwind_stressM[i, 1] - ϵ * abs(rwind_stressM[i, 1])

    sq_surface_uM = restimate_tracer_error(rmodelM, rwind_stressM)

    dsq_surface_u = (sq_surface_uP - sq_surface_uM) / diff

    @show ϵ, dsq_surface_u

end
