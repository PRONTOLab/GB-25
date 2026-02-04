using Reactant
using CUDA

using InteractiveUtils

using KernelAbstractions
using KernelAbstractions: @kernel, @index
using KernelAbstractions.NDIteration: NDIteration, NDRange, blocks, workitems, _Size

using KernelAbstractions: Kernel,
                          ndrange, workgroupsize,
                          __iterspace, __groupindex, __dynamic_checkbounds,
                          CompilerMetadata

using Base: @pure

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)


#
# Preliminary stuff for offsetstaticsize:
# TODO: when offsets are implemented in KA so that we can call `kernel(dev, group, size, offsets)`, remove all of this

import KernelAbstractions: partition
import KernelAbstractions: __ndrange, __groupsize
import KernelAbstractions: __validindex
import KernelAbstractions: get, expand, StaticSize

struct OffsetStaticSize{S} <: _Size
    function OffsetStaticSize{S}() where S
        new{S::Tuple{Vararg}}()
    end
end

@pure OffsetStaticSize(s::Tuple{Vararg{Int}}) = OffsetStaticSize{s}()
@pure OffsetStaticSize(s::Int...) = OffsetStaticSize{s}()
@pure OffsetStaticSize(s::Type{<:Tuple}) = OffsetStaticSize{tuple(s.parameters...)}()
@pure OffsetStaticSize(s::Tuple{Vararg{UnitRange{Int}}}) = OffsetStaticSize{s}()

# Some @pure convenience functions for `OffsetStaticSize` (following `StaticSize` in KA)
@pure get(::Type{OffsetStaticSize{S}}) where {S} = S
@pure get(::OffsetStaticSize{S}) where {S} = S
@pure Base.getindex(::OffsetStaticSize{S}, i::Int) where {S} = i <= length(S) ? S[i] : 1
@pure Base.ndims(::OffsetStaticSize{S}) where {S}  = length(S)
@pure Base.length(::OffsetStaticSize{S}) where {S} = prod(map(worksize, S))

@inline getrange(::OffsetStaticSize{S}) where {S} = worksize(S), offsets(S)
@inline getrange(::Type{OffsetStaticSize{S}}) where {S} = worksize(S), offsets(S)

# Makes sense to explicitly define the offsets for up to 3 dimensions,
# since Oceananigans typically runs kernels with up to 3 dimensions.
@inline offsets(ranges::NTuple{1, UnitRange}) = @inbounds (ranges[1].start - 1, )
@inline offsets(ranges::NTuple{2, UnitRange}) = @inbounds (ranges[1].start - 1, ranges[2].start - 1)
@inline offsets(ranges::NTuple{3, UnitRange}) = @inbounds (ranges[1].start - 1, ranges[2].start - 1, ranges[3].start - 1)

# Generic case for any number of dimensions
@inline offsets(ranges::NTuple{N, UnitRange}) where N = @inbounds Tuple(ranges[t].start - 1 for t in 1:N)

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

#
#
#

function my_interpolate_state!(arch, φᶠᶠᵃ)

    _my_launch!(arch,
            _my_interpolate_primary_atmospheric_state!,
            φᶠᶠᵃ)

end

# Inner interface
@inline function _my_launch!(dev, kernel!, first_kernel_arg, other_kernel_args...;
                          exclude_periphery = false,
                          reduced_dimensions = (),
                          active_cells_map = nothing)

    loop!, worksize = my_configure_kernel(dev, kernel!, active_cells_map, Val(exclude_periphery))

    # Don't launch kernels with no size
    if length(worksize) > 0
        loop!(first_kernel_arg, other_kernel_args...)
    end

    return nothing
end

# When there are KernelParameters, we use the `offset_work_layout` function
@inline function my_configure_kernel(dev, kernel!, thing, args...)

    workgroup = KernelAbstractions.NDIteration.StaticSize{(16, 16)}()
    worksize = OffsetStaticSize{(1:98, 1:50)}()
    

    loop = kernel!(dev, workgroup, worksize)

    return loop, worksize
end

@kernel function _my_interpolate_primary_atmospheric_state!(grid)

    i, j = @index(Global, NTuple)

    θ_degrees = my_rotation_angle(i, j, grid)
    sinθ = sind(θ_degrees)
end

@inline function my_rotation_angle(i, j, φᶠᶠᵃ)

    φᶠᶠᵃ⁺⁺ = φᶠᶠᵃ[i+1, j+1]
    φᶠᶠᵃ⁺⁻ = φᶠᶠᵃ[i+1, j]
    φᶠᶠᵃ⁻⁺ = φᶠᶠᵃ[i, j+1]
    φᶠᶠᵃ⁻⁻ = φᶠᶠᵃ[i, j]

    Δyᶠᶜᵃ⁺ = 2000.0
    Δyᶠᶜᵃ⁻ = 2000.0
    Δxᶜᶠᵃ⁺ = 1000.0
    Δxᶜᶠᵃ⁻ = 1000.0

    Rcosθ₁ = ifelse(Δyᶠᶜᵃ⁺ == 0, 0.0, deg2rad(φᶠᶠᵃ⁺⁺ - φᶠᶠᵃ⁺⁻) / Δyᶠᶜᵃ⁺)
    Rcosθ₂ = ifelse(Δyᶠᶜᵃ⁻ == 0, 0.0, deg2rad(φᶠᶠᵃ⁻⁺ - φᶠᶠᵃ⁻⁻) / Δyᶠᶜᵃ⁻)

    # θ is the rotation angle between intrinsic and extrinsic reference frame
    Rcosθ =   (Rcosθ₁ + Rcosθ₂) / 2
    Rsinθ = - (deg2rad(φᶠᶠᵃ⁺⁺ - φᶠᶠᵃ⁻⁺) / Δxᶜᶠᵃ⁺ + deg2rad(φᶠᶠᵃ⁺⁻ - φᶠᶠᵃ⁻⁻) / Δxᶜᶠᵃ⁻) / 2

    # Normalization for the rotation angles
    R = sqrt(Rcosθ^2 + Rsinθ^2)

    cosθ, sinθ = Rcosθ / R, Rsinθ / R

    θ_degrees = atand(sinθ / cosθ)
    return θ_degrees
end


@info "Compiling..."

dev = ReactantKernelAbstractionsExt.ReactantBackend()

#A = Float64.(reshape(1:(112*64), 112, 64))
#OA = OffsetArray(A, -7:104, -7:56)
#ROA = Reactant.to_rarray(OA)

B  = Float64.(reshape(1:(98*50), 98, 50))
RB = Reactant.to_rarray(B)

rfirst! = @compile raise=true sync=true my_interpolate_state!(dev, RB)

@info "Done!"
