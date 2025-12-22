using Reactant


using InteractiveUtils

using KernelAbstractions
using KernelAbstractions: @kernel, @index, Kernel
using KernelAbstractions.Extras.LoopInfo: @unroll
using OffsetArrays
using Base: @pure

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

using Oceananigans.Utils: OffsetStaticSize

#=
using CUDA: @device_override, blockIdx, threadIdx
using KernelAbstractions.NDIteration: _Size, StaticSize
using KernelAbstractions.NDIteration: NDRange

using KernelAbstractions.NDIteration
using KernelAbstractions: ndrange, workgroupsize

using KernelAbstractions: __iterspace, __groupindex, __dynamic_checkbounds
using KernelAbstractions: CompilerMetadata

import KernelAbstractions: partition
import KernelAbstractions: __ndrange, __groupsize
import KernelAbstractions: __validindex

struct OffsetStaticSize{S} <: _Size
    function OffsetStaticSize{S}() where S
        new{S::Tuple{Vararg}}()
    end
end

# Some @pure convenience functions for `OffsetStaticSize` (following `StaticSize` in KA)
@pure get(::Type{OffsetStaticSize{S}}) where {S} = S
@pure get(::OffsetStaticSize{S}) where {S} = S
@pure Base.getindex(::OffsetStaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::OffsetStaticSize{S}) where {S}  = length(S)
@pure Base.length(::OffsetStaticSize{S}) where {S} = prod(map(worksize, S))

@inline getrange(::OffsetStaticSize{S}) where {S} = worksize(S), offsets(S)
@inline getrange(::Type{OffsetStaticSize{S}}) where {S} = worksize(S), offsets(S)

@inline offsets(ranges::NTuple{N, UnitRange}) where N = Tuple(r.start - 1 for r in ranges)::NTuple{N}

@inline worksize(t::Tuple) = map(worksize, t)
@inline worksize(sz::Int) = sz
@inline worksize(r::AbstractUnitRange) = length(r)

const OffsetNDRange{N, S} = NDRange{N, <:StaticSize, <:StaticSize, <:Any, <:OffsetStaticSize{S}} where {N, S}

# NDRange has been modified to have offsets in place of workitems: Remember, dynamic offset kernels are not possible with this extension!!
# TODO: maybe don't do this
@inline function expand(ndrange::OffsetNDRange{N, S}, groupidx::CartesianIndex{N}, idx::CartesianIndex{N}) where {N, S}
    nI = ntuple(Val(N)) do I
        Base.@_inline_meta
        offsets = workitems(ndrange)
        stride = size(offsets, I)
        gidx = groupidx.I[I]
        (gidx - 1) * stride + idx.I[I] + S[I]
    end
    return CartesianIndex(nI)
end

@inline __ndrange(::CompilerMetadata{NDRange}) where {NDRange<:OffsetStaticSize}  = CartesianIndices(get(NDRange))
@inline __groupsize(cm::CompilerMetadata{NDRange}) where {NDRange<:OffsetStaticSize} = size(__ndrange(cm))

# Kernel{<:Any, <:StaticSize, <:StaticSize} and Kernel{<:Any, <:StaticSize, <:OffsetStaticSize} are the only kernels used by Oceananigans
const OffsetKernel = Kernel{<:Any, <:StaticSize, <:OffsetStaticSize}

# Extending the partition function to include offsets in NDRange: note that in this case the
# offsets take the place of the DynamicWorkitems which we assume is not needed in static kernels
function partition(kernel::OffsetKernel, inrange, ingroupsize)
    static_ndrange = ndrange(kernel)
    static_workgroupsize = workgroupsize(kernel)

    if inrange !== nothing && inrange != get(static_ndrange)
        error("Static NDRange ($static_ndrange) and launch NDRange ($inrange) differ")
    end

    range, offsets = getrange(static_ndrange)

    if static_workgroupsize <: StaticSize
        if ingroupsize !== nothing && ingroupsize != get(static_workgroupsize)
            error("Static WorkgroupSize ($static_workgroupsize) and launch WorkgroupSize $(ingroupsize) differ")
        end
        groupsize = get(static_workgroupsize)
    end

    @assert groupsize !== nothing
    @assert range !== nothing
    blocks, groupsize, dynamic = NDIteration.partition(range, groupsize)

    static_blocks = StaticSize{blocks}
    static_workgroupsize = StaticSize{groupsize} # we might have padded workgroupsize

    iterspace = NDRange{length(range), static_blocks, static_workgroupsize}(blocks, OffsetStaticSize(offsets))

    return iterspace, dynamic
end
=#
#
# End of needed utilities for OffsetStaticSize
#

function my_fill_west_and_east_halo!(c, dev, Hx, Nx)

    c_parent = parent.(c)

    _my_launch!(dev, my_fill_periodic_west_and_east_halo!, c_parent, Val(Hx), Nx)
    return nothing
end

@inline function _my_launch!(dev, kernel!, first_kernel_arg, other_kernel_args...)

    workgroup = KernelAbstractions.NDIteration.StaticSize{(16, 16)}()

    #@show @which OffsetStaticSize{(1:128, 1:32)}()
    worksize = OffsetStaticSize{(1:128, 1:32)}()

    loop! = kernel!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(first_kernel_arg, other_kernel_args...)

    return nothing
end

@kernel function my_fill_periodic_west_and_east_halo!(c, ::Val{H}, N) where {H}
    j, k = @index(Global, NTuple)
    ntuple(Val(length(c))) do n
        Base.@_inline_meta
        @unroll for i = 1:H
            @inbounds begin
                  c[n][i, j, k]     = c[n][N+i, j, k] # west
                  c[n][N+H+i, j, k] = c[n][H+i, j, k] # east
            end
        end
    end
end

Hx = 8
Nx = 112

rdev = ReactantKernelAbstractionsExt.ReactantBackend()
vdev = KernelAbstractions.CPU()

u = zeros(128, 128, 32)
vu = OffsetArray(u, -7:120, -7:120, -7:24)
ru = Reactant.to_rarray(vu)

vfields = (vu,)
rfields = (ru,)

@show typeof(vu), size(vu)
@show typeof(ru), size(ru)

opts = Reactant.CompileOptions(; max_constant_threshold=1000, raise=true)
#@show @code_hlo compile_options = opts my_fill_west_and_east_halo!(my_prognostic_fields(rmodel), rdev, Hx, Nx)

my_fill_west_and_east_halo!(vfields, vdev, Hx, Nx)
@jit compile_options = opts my_fill_west_and_east_halo!(rfields, rdev, Hx, Nx)

