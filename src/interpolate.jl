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
        mat = _resize_weight_mat(m, n)
        W = Reactant.Ops.constant(Tf.(mat))
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
    sample_f = ((0:(output_size-1)) .+ 0.5) .* inv_scale .- 0.5   # (output_size,)
    input_pos = 0.0:(input_size-1)                                  # (input_size,)
    x = Base.abs.(sample_f' .- input_pos)                           # (input_size, output_size)
    weights = max.(0.0, 1.0 .- x)
    col_sums = sum(weights; dims=1)
    weights = weights ./ col_sums
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
    end

    x = x isa Oceananigans.Field ? Reactant.ReactantCore.materialize_traced_array(Oceananigans.interior(x)) : x
    # Block (nearest-neighbor) upsample. Julia's `reshape` is column-major, so
    # to get block replication after the reshape we need the replication axes
    # to be the FAST-varying ones. Place src dims at output dims [2,4,6] and
    # the replication dims at [1,3,5] (shape [sx,nx,sy,ny,sz,nz]); then
    # reshape to (sx*nx, sy*ny, sz*nz) yields each src cell repeated sx×sy×sz
    # consecutively along each axis (= block-replicated upsample).
    x = Reactant.Ops.broadcast_in_dim(x, Int[2, 4, 6], Int[sx, nx, sy, ny, sz, nz])
    x = reshape(x, (sx * nx, sy * ny, sz * nz))

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

    x = summed / (sx * sy * sz)

    dst .= x
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Sharded-Reactant interpolate! pirate
# ═══════════════════════════════════════════════════════════════════════════════
#
# Type aliases for cross-architecture interpolation dispatch.

const ShardedDistributedArch     = Oceananigans.DistributedComputations.Distributed{<:Oceananigans.Architectures.ReactantState}
const ShardedGrid{FT,TX,TY,TZ}  = Oceananigans.Grids.AbstractGrid{FT,TX,TY,TZ,<:ShardedDistributedArch}
const ShardedField{LX,LY,LZ,O}  = Oceananigans.Fields.Field{LX,LY,LZ,O,<:ShardedGrid}

const CPUSourceGrid              = Oceananigans.Grids.AbstractGrid{<:Any,<:Any,<:Any,<:Any,<:Oceananigans.Architectures.CPU}
const CPUSourceField{LX,LY,LZ,O} = Oceananigans.Fields.Field{LX,LY,LZ,O,<:CPUSourceGrid}

function _gb25_interpolate_kernel!(to_field, from_field)
    to_grid   = to_field.grid
    from_grid = from_field.grid
    to_arch   = Oceananigans.Architectures.child_architecture(Oceananigans.Architectures.architecture(to_field))

    from_location = Tuple(L() for L in Oceananigans.Fields.location(from_field))
    to_location   = Tuple(L() for L in Oceananigans.Fields.location(to_field))

    params = Oceananigans.Utils.KernelParameters(Oceananigans.Fields.interior_indices(to_field))

    Oceananigans.Utils.launch!(to_arch, to_grid, params,
        Oceananigans.Fields._interpolate!, to_field, to_grid, to_location,
        from_field, from_grid, from_location)

    Oceananigans.BoundaryConditions.fill_halo_regions!(to_field)
    return to_field
end

function Oceananigans.Fields.interpolate!(target::ShardedField, source::CPUSourceField)
    compiled = @compile sync = true raise = true _gb25_interpolate_kernel!(target, source)
    compiled(target, source)
    Reactant.XLA.IFRT.free_exec(compiled.exec)
    compiled.exec.exec = C_NULL
    return target
end
