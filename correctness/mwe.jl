using CUDA
using KernelAbstractions
using Reactant

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

const ReactantBackend = ReactantKernelAbstractionsExt.ReactantBackend

@kernel function _integral!(p, Nz, b)
    i, j = @index(Global, NTuple)
    @inbounds p[i, j, Nz] = b[i, j, Nz] + b[i, j, Nz+1]

    for k in Nz-1 : -1 : 1
        @inbounds p[i, j, k] = p[i, j, k+1] + b[i, j, k] + b[i, j, k+1]
    end
end

function problem_kernel!(p, b)
    backend = if p isa Array
        CPU()
    else
        ReactantBackend()
    end

    Nz = size(p, 3) - 1
    Nx, Ny, _ = size(p)

    Wx = min(16, Nx)
    Wy = min(16, Ny)
    workgroup = (Wx, Wy)

    workgroup = (Wx, Wy)
    worksize = (Nx, Ny, Nz)
    loop! = _integral!(backend, workgroup, worksize)
    loop!(p, Nz, b)

    return nothing
end

raise = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory" #,raise-affine-to-stablehlo{prefer_while_raising=false dump_failed_lockstep=true},canonicalize,arith-raise{stablehlo=true}"

Nx = 16
Ny = 32
Nz = 8
p = zeros(Nx, Ny, Nz+1)
b = rand(Nx, Ny, Nz+1)

rp = Reactant.to_rarray(p)
rb = Reactant.to_rarray(b)

δb = b .- Array(rb)
@info "input"
@show isapprox(b, Array(rb), rtol=1e-3)

# Also tests that kernel isn't bugged
problem_kernel!(p, b)

# Works: 
#rproblem! = @compile sync=true raise=raise problem_kernel!(rp, rb)

# Doesn't work: 
rproblem! = @compile sync=true raise=true problem_kernel!(rp, rb)
rproblem!(rp, rb)

δp = p .- Array(rp)
@info "output"
@show isapprox(p, Array(rp), rtol=1e-3)

