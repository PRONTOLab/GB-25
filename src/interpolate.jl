using Reactant
using Reactant: Ops, Ops.@opcall

@noinline function resize_linear(
    src::Reactant.TracedRArray{T,N}, shape
) where {T,N}
    output_shape = shape
    Tf = float(T)
    x = src
    for d in 1:N
        size(x, d) == output_shape[d] && continue
        m = size(x, d)
        n = output_shape[d]
        # Weight matrix: (m, n). Computed at trace time from static sizes.
        mat = _resize_weight_mat(m, n)
        W = Reactant.Ops.constant(Tf.(mat))
        # Move dim d to position 1, flatten remaining dims, matmul, restore.
        perm = [d; setdiff(1:ndims(x), d)]
        xt = permutedims(x, perm)          # (m, d2, d3, ...)
        rest = size(xt)[2:end]
        xt = @opcall reshape(xt, m, prod(rest))    # (m, rest_flat)
        W′ = @opcall transpose(W, collect(Int64, ndims(W):-1:1))
        xt = @opcall dot_general(W′, xt; contracting_dimensions=([2], [1]))  # (n, rest_flat)
        xt = @opcall reshape(xt, n, rest...)
        x = permutedims(xt, invperm(perm))
    end

    return x
end

function _resize_weight_mat(input_size::Int, output_size::Int)
    inv_scale = input_size / output_size
    # Center of each output pixel in 0-indexed input-pixel coordinates
    sample_f = ((0:(output_size-1)) .+ 0.5) .* inv_scale .- 0.5   # (output_size,)
    input_pos = 0.0:(input_size-1)                                  # (input_size,)
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

function interpolate_3d!(dst::AbstractArray{T1,3}, x::AbstractArray{T2,3}) where {T1,T2}
    nx, ny, nz = size(x)
    dx, dy, dz = size(dst)
    sx, sy, sz = dx ÷ nx, dy ÷ ny, dz ÷ nz

    if sx * nx != dx || sy * ny != dy || sz * nz != dz
        @warn "interpolate: invalid factors, skipping interpolate" sx sy sz size(x) size(dst)
        return nothing

        dst .=
            resize_linear(
                Reactant.ReactantCore.materialize_traced_array(Oceananigans.interior(x)),
                (dx, dy, dz)
            )

        return nothing
    end

    #    x = jnp.expand_dims(data, (1, 3, 5))
    #    x = reshape(x, (1, 3, 5))
    #    x = jnp.broadcast_to(x, (nx, sx, ny, sy, nz, sz))
    x = x isa Oceananigans.Field ? Reactant.ReactantCore.materialize_traced_array(Oceananigans.interior(x)) : x
    # Block (nearest-neighbor) upsample. Julia's `reshape` is column-major, so
    # to get block replication after the reshape we need the replication axes
    # to be the FAST-varying ones. Place src dims at output dims [2,4,6] and
    # the replication dims at [1,3,5] (shape [sx,nx,sy,ny,sz,nz]); then
    # reshape to (sx*nx, sy*ny, sz*nz) yields each src cell repeated sx×sy×sz
    # consecutively along each axis (= block-replicated upsample).
    x = Reactant.Ops.broadcast_in_dim(x, Int[2, 4, 6], Int[sx, nx, sy, ny, sz, nz])
    x = reshape(x, (sx * nx, sy * ny, sz * nz))

    #    summed = lax.reduce_window(
    #        x,
    #        0.0,
    #        lax.add,
    #        window_dimensions=scales,
    #        window_strides=(1, 1, 1),
    #        padding=((0, sx - 1), (0, sy - 1), (0, sz - 1))
    #    )
    summed = Reactant.Ops.reduce_window(
        +,
        [Reactant.ReactantCore.materialize_traced_array(x)],
        [Reactant.Ops.constant(zero(T2))];
        window_dimensions=Int64[sx, sy, sz],
        window_strides=ones(Int64, 3),
        window_dilations=ones(Int64, 3),
        base_dilations=ones(Int64, 3),
        padding_low=zeros(Int64, 3),
        padding_high=Int64[sx, sy, sz] .- 1,
        output_shape=Int64[nx*sx, ny*sy, nz*sz]
    ) |> only

    #    # 3) Normalize by window area (2 * 3 * 4 = 24)
    #    return summed / (sx * sy * sz)
    x = summed / (sx * sy * sz)

    dst .= x
    return nothing
end

#    nx, ny, nz = data.shape
#    sx, sy, sz = scales
#    
#    # 1) Nearest neighbor upsample using broadcast and reshape
#    x = jnp.expand_dims(data, (1, 3, 5))
#    x = jnp.broadcast_to(x, (nx, sx, ny, sy, nz, sz))
#    x = jnp.reshape(x, (nx * sx, ny * sy, nz * sz))
#    
#    # 2) Moving average via reduce_window to get linear interpolation
#    # Padding at the end by (scale - 1) makes the output maintain the 
#    # exact shape of nx*sx, ny*sy, nz*sz while computing moving sums.
#    summed = lax.reduce_window(
#        x,
#        0.0,
#        lax.add,
#        window_dimensions=scales,
#        window_strides=(1, 1, 1),
#        padding=((0, sx - 1), (0, sy - 1), (0, sz - 1))
#    )
#    
#    # 3) Normalize by window area (2 * 3 * 4 = 24)
#    return summed / (sx * sy * sz)

