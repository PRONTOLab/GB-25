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
# Source array preparation for FaceInterpolateArray
# ═══════════════════════════════════════════════════════════════════════════════
#
# Prepares a raw source array (no halos) for FaceInterpolateArray by adding
# halo padding: periodic wrapping for x, zero-gradient for y/z.
#
# The source is NOT trimmed — face+bounded dims retain all N+1 faces so that
# boundary values are available for interpolation near the domain edge.
# FaceInterpolateArray uses `bounded_dims` to adjust the scale factor instead.

function prepare_source_for_interpolation(
    src_array::AbstractArray{<:Any,3}, halo::NTuple{3,Int},
    face_dims::NTuple{3,Bool}, ::Type{FT},
) where {FT}
    data = FT.(src_array)

    nx, ny, nz = size(data)
    Hx, Hy, Hz = halo
    padded = zeros(FT, nx + 2Hx, ny + 2Hy, nz + 2Hz)

    padded[Hx+1:Hx+nx, Hy+1:Hy+ny, Hz+1:Hz+nz] .= data

    padded[1:Hx,          Hy+1:Hy+ny, Hz+1:Hz+nz] .= data[end-Hx+1:end, :, :]
    padded[Hx+nx+1:end,   Hy+1:Hy+ny, Hz+1:Hz+nz] .= data[1:Hx, :, :]

    for h in 1:Hy
        padded[:, h, Hz+1:Hz+nz]      .= padded[:, Hy+1, Hz+1:Hz+nz]
        padded[:, end-h+1, Hz+1:Hz+nz] .= padded[:, end-Hy, Hz+1:Hz+nz]
    end

    for h in 1:Hz
        padded[:, :, h]        .= padded[:, :, Hz+1]
        padded[:, :, end-h+1]  .= padded[:, :, end-Hz]
    end

    return padded
end

# ═══════════════════════════════════════════════════════════════════════════════
# Face-aware InterpolateArray
# ═══════════════════════════════════════════════════════════════════════════════
#
# Drop-in replacement for Reactant.InterpolateArray that accepts `face_dims`
# and `bounded_dims` tuples.
#
# face_dims:   dimensions where the field is Face-staggered
# bounded_dims: dimensions with Bounded topology (y, z on LatLonGrid)
#
# For face+bounded dims the source retains all N+1 faces (not trimmed), so the
# scale factor uses (M_interior - 1) to get the correct cells-ratio mapping:
#     fi = (I − 1) × (M−1)/N + 1      face+bounded
#     fi = (I − 1) × M/N + 1           face+periodic
#     fi = (I − 0.5) × M/N + 0.5       center (any topology)
#
# Linear interpolation accumulates in Float64 to avoid precision loss on large
# grids where total_den can exceed 9e8.

function FaceInterpolateArray(
    local_cpu_array::AbstractArray{T,N},
    final_grid_size::NTuple{N,Int},
    sharding::Reactant.Sharding.AbstractSharding,
    interpolation::Reactant.InterpolationType.T,
    halo::NTuple{N,Int} = ntuple(_ -> 0, N);
    face_dims::NTuple{N,Bool} = ntuple(_ -> false, N),
    bounded_dims::NTuple{N,Bool} = ntuple(_ -> false, N),
    client = nothing,
) where {T,N}
    @assert Reactant.Sharding.is_sharded(sharding)
    client = client === nothing ? Reactant.XLA.default_backend() : client
    (; hlo_sharding) = Reactant.Sharding.HloSharding(sharding, final_grid_size)
    all_devices = Reactant.XLA.get_device.((client,), sharding.mesh.device_ids)

    addressable_device_indices = [
        i - 1 for (i, device) in enumerate(all_devices) if Reactant.XLA.is_addressable(device)
    ]

    addressable_slices, _ = Reactant.XLA.sharding_to_concrete_array_indices(
        hlo_sharding, final_grid_size, addressable_device_indices
    )
    src_size = size(local_cpu_array)
    ordered_buffers = Vector{Array{T,N}}(undef, length(addressable_slices))

    for (buf_idx, slice) in enumerate(addressable_slices)
        shard_shape = length.(slice)

        if interpolation == Reactant.InterpolationType.Nearest
            src_idx_ranges = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]
                is_face = face_dims[dim]
                is_bounded = bounded_dims[dim]
                M_dim_shifted = M_dim - 2*H
                effective_M = (is_face && is_bounded) ? M_dim_shifted - 1 : M_dim_shifted

                [begin
                    if I <= H
                        clamp(I, 1, M_dim)
                    elseif I >= N_dim - H + 1
                        clamp(M_dim - N_dim + I, 1, M_dim)
                    else
                        I_shifted = I - H
                        N_dim_shifted = N_dim - 2*H

                        if N_dim_shifted <= 0 || effective_M <= 0
                            clamp(I, 1, M_dim)
                        else
                            idx_shifted = if is_face
                                ((I_shifted - 1) * effective_M + N_dim_shifted - 1) ÷ N_dim_shifted + 1
                            else
                                (I_shifted * effective_M + N_dim_shifted - 1) ÷ N_dim_shifted
                            end
                            clamp(idx_shifted + H, 1, M_dim)
                        end
                    end
                end for I in I_range]
            end

            buf = Array{T,N}(undef, shard_shape)
            for I in CartesianIndices(shard_shape)
                idx = ntuple(dim -> src_idx_ranges[dim][I.I[dim]], N)
                buf[I] = local_cpu_array[CartesianIndex(idx)]
            end

        elseif interpolation == Reactant.InterpolationType.Linear
            lows = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]
                is_face = face_dims[dim]
                is_bounded = bounded_dims[dim]
                M_dim_shifted = M_dim - 2*H
                effective_M = (is_face && is_bounded) ? M_dim_shifted - 1 : M_dim_shifted

                [begin
                    if I <= H
                        clamp(I, 1, M_dim)
                    elseif I >= N_dim - H + 1
                        clamp(M_dim - N_dim + I, 1, M_dim)
                    else
                        I_shifted = I - H
                        N_dim_shifted = N_dim - 2*H

                        a = if is_face
                            2 * (I_shifted - 1) * effective_M + 2 * N_dim_shifted
                        else
                            (2 * I_shifted - 1) * effective_M + N_dim_shifted
                        end
                        b = 2 * N_dim_shifted
                        low_shifted = a ÷ b
                        clamp(low_shifted + H, 1, M_dim)
                    end
                end for I in I_range]
            end

            highs = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]
                is_face = face_dims[dim]
                is_bounded = bounded_dims[dim]
                M_dim_shifted = M_dim - 2*H
                effective_M = (is_face && is_bounded) ? M_dim_shifted - 1 : M_dim_shifted

                [begin
                    if I <= H
                        clamp(I, 1, M_dim)
                    elseif I >= N_dim - H + 1
                        clamp(M_dim - N_dim + I, 1, M_dim)
                    else
                        I_shifted = I - H
                        N_dim_shifted = N_dim - 2*H

                        a = if is_face
                            2 * (I_shifted - 1) * effective_M + 2 * N_dim_shifted
                        else
                            (2 * I_shifted - 1) * effective_M + N_dim_shifted
                        end
                        b = 2 * N_dim_shifted
                        low_shifted = a ÷ b
                        clamp(low_shifted + 1 + H, 1, M_dim)
                    end
                end for I in I_range]
            end

            dens = ntuple(N) do dim
                H = halo[dim]
                2 * max(1, final_grid_size[dim] - 2 * H)
            end
            total_den = prod(dens)

            rems = ntuple(N) do dim
                I_range = slice[dim]
                N_dim, M_dim = final_grid_size[dim], src_size[dim]
                H = halo[dim]
                is_face = face_dims[dim]
                is_bounded = bounded_dims[dim]
                M_dim_shifted = M_dim - 2*H
                effective_M = (is_face && is_bounded) ? M_dim_shifted - 1 : M_dim_shifted

                [begin
                    if I <= H || I >= N_dim - H + 1
                        0
                    else
                        I_shifted = I - H
                        N_dim_shifted = N_dim - 2*H

                        a = if is_face
                            2 * (I_shifted - 1) * effective_M + 2 * N_dim_shifted
                        else
                            (2 * I_shifted - 1) * effective_M + N_dim_shifted
                        end
                        b = 2 * N_dim_shifted
                        a % b
                    end
                end for I in I_range]
            end

            buf = Array{T,N}(undef, shard_shape)
            corner_space = CartesianIndices(ntuple(_ -> 2, N))
            for I in CartesianIndices(shard_shape)
                sum_val = zero(Float64)
                for c in corner_space
                    idx = ntuple(
                        dim -> c[dim] == 1 ? lows[dim][I.I[dim]] : highs[dim][I.I[dim]], N
                    )
                    w_int = prod(
                        ntuple(
                            dim -> if c[dim] == 1
                                (dens[dim] - rems[dim][I.I[dim]])
                            else
                                rems[dim][I.I[dim]]
                            end,
                            N,
                        ),
                    )
                    sum_val += w_int * Float64(local_cpu_array[CartesianIndex(idx)])
                end
                buf[I] = T(sum_val / total_den)
            end
        else
            error("Unsupported interpolation type")
        end
        ordered_buffers[buf_idx] = buf
    end
    return Reactant.ConcreteIFRTArray(ordered_buffers, final_grid_size; client, sharding)
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
