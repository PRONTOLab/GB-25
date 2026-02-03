using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init
using Reactant

using InteractiveUtils

using Oceananigans

using KernelAbstractions
using KernelAbstractions: @kernel, @index

using OffsetArrays

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

@info "Generating model..."

GC.gc(true); GC.gc(false); GC.gc(true)

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
    worksize = Oceananigans.Utils.OffsetStaticSize{(0:97, 0:49)}()
    

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

A = Float64.(reshape(1:(112*64), 112, 64))
OA = OffsetArray(A, -7:104, -7:56)
ROA = Reactant.to_rarray(OA)

rfirst! = @compile raise=true sync=true my_interpolate_state!(dev, ROA)

@info "Done!"
