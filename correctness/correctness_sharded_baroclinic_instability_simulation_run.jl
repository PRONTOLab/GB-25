ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans
using Reactant

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

Rx, Ry = 2, 2

H = 8
Tx = 64 * Rx
Ty = 64 * Ry
Nz = 16

Nx = Tx - 2H
Ny = Ty - 2H

using InteractiveUtils
using KernelAbstractions
using KernelAbstractions: @kernel, @index

using OffsetArrays

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

function my_compute_hydrostatic_momentum_tendencies!(u, dev, Ny, Nz)

    loop!, worksize = my_configure_kernel(dev, my_compute_hydrostatic_free_surface_Gu!)

    loop!(u, Ny, Nz)

    return nothing
end

@inline function my_configure_kernel(dev, kernel!)

    workgroup = (16, 16)
    worksize = (112, 112, 16)

    loop = kernel!(dev, workgroup, worksize)

    return loop, worksize
end

@kernel function my_compute_hydrostatic_free_surface_Gu!(Gu, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    active_nodes = (!((j-1 < 1) | (j-1 > Ny))
                  + !( (j+1 > Ny) ))

    @inbounds Gu[i, j, k] = active_nodes
end

U  = zeros(Float64, 128, 128, 32)
vU = OffsetArray(U, -7:120, -7:120, -7:24)
rU = Reactant.to_rarray(vU)

# correct
opts = Reactant.CompileOptions(; max_constant_threshold=10, raise=true)
# incorrect
optimization_passes = """
mark-func-memory-effects{assume_no_memory_effects=false max_iterations=8},inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },propagate-constant-bounds,sroa-wrappers{attributor=true dump_postllvm=false dump_prellvm=false instcombine=false instsimplify=true set_private=true sroa=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},sroa-wrappers{attributor=true dump_postllvm=false dump_prellvm=false instcombine=false instsimplify=true set_private=true sroa=true},libdevice-funcs-raise{remove_freeze=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},remove-duplicate-func-def,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-hlo-generate-td{create-module=false flags= patterns=compare_op_canon<16>;transpose_transpose<16>;broadcast_in_dim_op_canon<16>;convert_op_canon<16>;dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;chained_dynamic_broadcast_in_dim_canonicalization<16>;dynamic_broadcast_in_dim_all_dims_non_expanding<16>;noop_reduce_op_canon<16>;empty_reduce_op_canon<16>;dynamic_reshape_op_canon<16>;get_tuple_element_op_canon<16>;real_op_canon<16>;imag_op_canon<16>;conj_complex_negate<16>;get_dimension_size_op_canon<16>;reshape_op_canon<16>;merge_consecutive_reshapes<16>;transpose_is_reshape<16>;zero_extent_tensor_canon<16>;cse_broadcast_in_dim<16>;cse_slice<16>;cse_transpose<16>;cse_convert<16>;cse_dot_general<16>;cse_reshape<16>;cse_mul<16>;cse_div<16>;cse_add<16>;cse_subtract<16>;cse_min<16>;cse_max<16>;cse_neg<16>;cse_abs<16>;cse_concatenate<16>;concatenate_op_canon<16>(1000);select_op_canon<16>(1000);add_simplify<16>;sub_simplify<16>;and_simplify<16>;max_simplify<16>;min_simplify<16>;or_simplify<16>;xor_simplify<16>;mul_simplify<16>;div_simplify<16>;rem_simplify<16>;pow_simplify<16>;simplify_extend<16>;simplify_wrap<16>;simplify_rotate<16>;noop_slice<16>;noop_reverse<16>;slice_slice<16>;dynamic_slice_slice<16>;slice_dynamic_slice<16>;dynamic_slice_dynamic_slice<16>;shift_right_logical_simplify<16>;slice_simplify<16>;convert_simplify<16>;dynamic_slice_to_static<16>;dynamic_update_slice_elim<16>;concat_to_broadcast<16>;reduce_to_reshape<16>;broadcast_to_reshape<16>;slice_internal;iota_simplify<16>(1000);broadcast_in_dim_simplify<16>(1000);convert_concat<1>;dynamic_update_to_concat<1>;slice_of_dynamic_update<1>;slice_elementwise<1>;dot_reshape_dot<1>;concat_fuse<1>;concat_push_binop_add<1>;concat_push_binop_mul<1>;reduce_concat<1>;slice_concat<1>;concat_slice<1>;select_op_used_within_if<1>;bin_broadcast_splat_add<1>;bin_broadcast_splat_subtract<1>;bin_broadcast_splat_div<1>;bin_broadcast_splat_mul<1>;dot_general_simplify<16>;transpose_simplify<16>;reshape_empty_broadcast<1>;broadcast_reshape<1>;transpose_dot_reorder<1>;dot_transpose<1>;transpose_convolution<1>;convolution_transpose<1>;convert_convert_float<1>;convert_convert_int<1>;reshape_iota<1>;broadcast_reduce<1>;slice_dot_general<1>;if_inline<1>;if_to_select<1>;divide_sqrt_to_multiply_rsqrt<16>;associative_binary_op_reordering<1>;transpose_broadcast_in_dim_to_broadcast_in_dim<16>;replace_neg_add_with_subtract;replace_subtract_neg_with_add;binop_const_simplify;not_select_simplify;common_compare_expression_rewrite;compare_select_simplify;while_simplify<1>(1);if_remove_unused;transpose_reshape_to_broadcast;reshape_transpose_to_broadcast;dus_dus;dus_dus_concat;abs_positive_simplify;transpose_elementwise_transpose;select_comp_iota_const_simplify<1>;sign_abs_simplify<1>;broadcastindim_is_reshape;slice_reduce_window<1>;while_deadresult;while_dus;while_op_induction_replacement;dus_concat;slice_dus_to_concat;while_induction_reduction;slice_broadcast;associative_common_mul_op_reordering;slice_select_to_select_slice;slice_if;dus_to_i32;slice_extend;concat_wrap;cse_extend<16>;cse_wrap<16>;cse_rotate<16>;cse_rotate<16>;concat_concat_axis_swap;concat_concat_to_dus;broadcast_iota_simplify;select_comp_iota_to_dus;compare_cleanup;broadcast_compare;not_compare;broadcast_iota;cse_iota;compare_iota_const_simplify radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,lower-kernel{backend=cpu},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},llvm-to-memref-access,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},convert-llvm-to-cf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},enzyme-lift-cf-to-scf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(canonicalize-loops),canonicalize-scf-for,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},libdevice-funcs-raise{remove_freeze=true},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},affine-cfg,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(canonicalize-loops),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},llvm-to-affine-access,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},delinearize-indexing,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},simplify-affine-exprs,affine-cfg,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(affine-loop-invariant-code-motion),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},sort-memory,raise-affine-to-stablehlo{dump_failed_lockstep=false enable_lockstep_for=true err_if_not_fully_raised=true prefer_while_raising=false},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},arith-raise{stablehlo=true},enzyme-hlo-generate-td{create-module=false flags= patterns=iota_simplify<16>(1000) radix=10},transform-interpreter{ debug-payload-root-tag= disable-expensive-checks=false entry-point=__transform_main},enzyme-hlo-remove-transform,symbol-dce
"""
opts = Reactant.CompileOptions(; max_constant_threshold=1000, raise=true, optimization_passes)

@show @code_hlo compile_options = opts my_compute_hydrostatic_momentum_tendencies!(rU, ReactantKernelAbstractionsExt.ReactantBackend(), Ny, Nz)

@jit compile_options = opts my_compute_hydrostatic_momentum_tendencies!(rU, ReactantKernelAbstractionsExt.ReactantBackend(), Ny, Nz)
my_compute_hydrostatic_momentum_tendencies!(vU, KernelAbstractions.CPU(), Ny, Nz)

@info "After initialization and update state (should be 0, or at most maybe machine precision, but there's a bug):"
rU  = convert(Array, parent(rU))
vU  = parent(vU)

@show maximum(abs.(rU - vU))
