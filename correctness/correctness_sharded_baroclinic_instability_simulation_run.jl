ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans
using Reactant

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
opts = Reactant.CompileOptions(; max_constant_threshold=1000, raise=true)

@show @code_hlo compile_options = opts my_compute_hydrostatic_momentum_tendencies!(rU, ReactantKernelAbstractionsExt.ReactantBackend(), Ny, Nz)

@jit compile_options = opts my_compute_hydrostatic_momentum_tendencies!(rU, ReactantKernelAbstractionsExt.ReactantBackend(), Ny, Nz)
my_compute_hydrostatic_momentum_tendencies!(vU, KernelAbstractions.CPU(), Ny, Nz)

@info "After initialization and update state (should be 0, or at most maybe machine precision, but there's a bug):"
rU  = convert(Array, parent(rU))
vU  = parent(vU)

@show maximum(abs.(rU - vU))
