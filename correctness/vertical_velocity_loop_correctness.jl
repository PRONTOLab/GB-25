using GordonBell25
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using Reactant

H = 8 # halo size
Tx, Ty = 128, 64
Nx, Ny = (Tx, Ty) .- 2H
Nz = 32

kw = (
    resolution = 2,
    free_surface = nothing, # SplitExplicitFreeSurface(substeps=10),
    coriolis = nothing,
    buoyancy = BuoyancyTracer(),
    closure = nothing, #vertical_diffusivity,
    momentum_advection = nothing, #WENOVectorInvariant(),
    tracer_advection = nothing, #WENO(),
    Δt = 60,
    Nz = 10,
)

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(); kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(); kw...)

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)

@jit Oceananigans.initialize!(rmodel)
Oceananigans.initialize!(vmodel)

@jit Oceananigans.TimeSteppers.update_state!(rmodel)
Oceananigans.TimeSteppers.update_state!(vmodel)

GordonBell25.sync_states!(rmodel, vmodel)
GordonBell25.compare_states(rmodel, vmodel)

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)

@time rfirst!(rmodel)
@time GordonBell25.first_time_step!(vmodel)

# Everything is kind of correct till here (errors of about 1e-14)

GordonBell25.compare_states(rmodel, vmodel)

# Runs with this:
# passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf" 

# Runs with this:
# passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing" 

# Runs with this:
# passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg" 

# Runs with this:
# passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory" 

# Does not run with this:
passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-affine-to-stablehlo{prefer_while_raising=false dump_failed_lockstep=true},canonicalize,arith-raise{stablehlo=true}" 

Nt  = 3
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=passes GordonBell25.loop!(rmodel, rNt)
@time rloop!(rmodel, rNt)
@time GordonBell25.loop!(vmodel, Nt)

# Correctness does not work on loops apparently (only for w)

GordonBell25.compare_states(rmodel, vmodel)

