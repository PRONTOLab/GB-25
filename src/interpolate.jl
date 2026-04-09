using Reactant

"""
    resize_linear!(dst::AbstractArray{T,N}, src::AbstractArray{T,N}, output_shape::NTuple{N,Int}) -> AbstractArray

The interpolation follows JAX's `jax.image.resize` with `ResizeMethod.LINEAR`: for each
spatial dimension a weight matrix is computed via a triangle kernel and applied via
matrix multiplication. The weight matrix depends only on the input/output sizes (which
are static during JIT compilation), so it becomes an XLA constant with no runtime
overhead.

# Examples
```julia
x = rand(Float32, 4, 4)
y = resize(x, (8, 8))   # 2× upsample
z = resize(x, (2, 3))   # mixed downsample / upsample
```
"""
@noinline function resize_linear!(
    dst::AbstractArray{T,N}, src::AbstractArray{T,N}
) where {T,N}
    output_shape = size(dst)
    Tf = float(T)
    x = Tf.(src)
    for d in 1:N
        size(x, d) == output_shape[d] && continue
        m = size(x, d)
        n = output_shape[d]
        # Weight matrix: (m, n). Computed at trace time from static sizes.
        mat = _resize_weight_mat(m, n)
        W = Tf.(Reactant.Ops.constant(mat))
        # Move dim d to position 1, flatten remaining dims, matmul, restore.
        perm = [d; setdiff(1:ndims(x), d)]
        xt = permutedims(x, perm)          # (m, d2, d3, ...)
        rest = size(xt)[2:end]
        xt = reshape(xt, m, prod(rest))    # (m, rest_flat)
        W′ = Reactant.Ops.transpose(W, collect(Int64, ndims(W):-1:1))
        xt = Reactant.Ops.dot_general(W′, xt; contracting_dimensions=([2], [1]))  # (n, rest_flat)
        xt = Reactant.Ops.reshape(xt, n, rest...)
        x = permutedims(xt, invperm(perm))
    end
    return copyto!(dst, x)
end

function _resize_weight_mat(input_size::Int, output_size::Int)
    inv_scale = input_size / output_size
    # Center of each output pixel in 0-indexed input-pixel coordinates
    sample_f = ((0:(output_size - 1)) .+ 0.5) .* inv_scale .- 0.5   # (output_size,)
    input_pos = 0.0:(input_size - 1)                                  # (input_size,)
    # x[i, j] = |sample_f[j] - (i-1)|; broadcasting over both dims
    x = Base.abs.(sample_f' .- input_pos)                                  # (input_size, output_size)
    # Triangle kernel: max(0, 1 - |x|)
    weights = max.(0.0, 1.0 .- x)
    # Normalize each column
    col_sums = sum(weights; dims=1)
    weights = weights ./ col_sums
    # Zero out columns where the sample is outside the valid input range
    valid = (sample_f .>= -0.5) .& (sample_f .<= input_size - 0.5)
    weights = weights .* valid'
    return weights   # (input_size, output_size), Float64
end

"""
    resize(x::AbstractArray{T,N}, output_shape::NTuple{N,Int}) -> AbstractArray

Resize array `x` to `output_shape` using linear interpolation. See [`resize_linear!`](@ref).
"""
function resize(x::AbstractArray{T,N}, output_shape::NTuple{N,Int}) where {T,N}
    dst = similar(x, float(T), output_shape)
    return resize_linear!(dst, float(T).(x))
end

