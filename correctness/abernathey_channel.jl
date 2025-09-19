ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Reactant
using GordonBell25
using KernelAbstractions: @kernel, @index, StaticSize, CPU

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

const ReactantBackend = ReactantKernelAbstractionsExt.ReactantBackend

function wrapper!(arch, Gv, z)
    workgroup = (16, 16)
    worksize  = (8, 8, 8)
    dev       = arch
    loop!     = _add_ridge_check!(dev, workgroup, worksize)

    # Don't launch kernels with no size    
    loop!(Gv, z)

    return nothing
end

""" Calculate the right-hand-side of the v-velocity equation. """
@kernel function _add_ridge_check!(Gv, z)
    i, j, k = @index(Global, NTuple)
    @inbounds Gv[i, j, k] = !(z[k] ≤ -1500)
end

#####
##### Actually creating our model and using these functions to run it:
#####

@info "Vanilla model as a comparison..."

# "Vanilla" data:
varch = CPU()
vGv   = zeros((8, 9, 8))
vz    = [-185.99366913029345, -166.00501682133535, -146.0163645123772, -126.0277122034191, -106.03905989446096, -86.99190207831609, -69.73904173760515, -54.111450849280025,
         -39.95602432000002, -27.13408000000002, -15.520000000000017, -5.000000000000007, 5.000000000000007, 15.000000000000021, 25.000000000000036, 35.00000000000005]

# Architecture
arch = ReactantBackend()
Gv   = Reactant.ConcreteRArray(vGv)
z    = Reactant.ConcreteRArray(vz)

@info "Built model."

using InteractiveUtils

@info "Comparing the pre-run model states..."

@allowscalar @show maximum(abs.(convert(Array, Gv) - convert(Array, vGv)))

@info "Running the vanilla model"
tic       = time()
wrapper!(varch, vGv, vz)
vrun_toc  = time() - tic
@show vrun_toc

@info "Compiling the model run..."
tic = time()
thing = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-affine-to-stablehlo{prefer_while_raising=false},canonicalize,arith-raise{stablehlo=true}"
rwrapper! = @compile raise_first=true raise=thing sync=true wrapper!(arch, Gv, z)
compile_toc = time() - tic

@show compile_toc

@info "Running the Reactant model"

tic      = time()
rwrapper!(arch, Gv, z)
rrun_toc = time() - tic
@show rrun_toc

@info "Comparing the model states after running..."

@allowscalar @show maximum(abs.(convert(Array, Gv) - convert(Array, vGv)))