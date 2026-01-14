using Reactant
using CUDA
using InteractiveUtils
using KernelAbstractions: @kernel, @index

function my_compute_tendencies!(c_tendency, dev, Nz)

    _my_launch!(dev,
            my_compute_hydrostatic_free_surface_Gc!,
            c_tendency,
            Nz)

    return nothing
end

@inline function _my_launch!(dev, kernel!, first_kernel_arg, other_kernel_args...)

    workgroup = (16, 16)
    worksize  = (48, 24, 10)

    loop! = kernel!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(first_kernel_arg, other_kernel_args...)

    return nothing
end

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function my_compute_hydrostatic_free_surface_Gc!(Gc, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = ifelse(my_outside_biased_halo_zᶠ(k, Nz), 100.0, 3.0)
end

@inline my_outside_biased_halo_zᶠ(i, N) = (i >= 3 + 1) & (i <= N + 1 - (3 - 1)) &  # Left bias
                                          (i >= 3)     & (i <= N + 1 - 3)          # Right bias

@info "Compiling..."

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

dev = ReactantKernelAbstractionsExt.ReactantBackend()
Nz  = 10

data = ones(64, 40, 26)
rdata = Reactant.to_rarray(data)

rfirst! = @compile raise=true sync=true my_compute_tendencies!(rdata, dev, Nz)

#my_compute_tendencies!(model)
