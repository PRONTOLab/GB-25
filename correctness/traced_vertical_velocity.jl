using GordonBell25
using KernelAbstractions: @index, @kernel
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using Reactant
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, div_xyᶜᶜᶜ, Δzᶜᶜᶜ, Azᶜᶜᶜ

@kernel function _compute_w!(w, u, v, Nz)
    i, j = @index(Global, NTuple)
    @inbounds w[i, j, 1] = 0
    for k in 2:Nz+1
        @inbounds begin
            δ = u[i+1, j, k] - u[i, j, k] +
                v[i, j+1, k] - v[i, j, k]

            w[i, j, k] = w[i, j, k-1] - δ
        end
    end
end

compute_w!(w, u, v, arch, grid) = Oceananigans.Utils.launch!(arch, grid, :xy, _compute_w!, w, u, v, grid.Nz)

kw = (
    resolution = 4,
    free_surface = ExplicitFreeSurface(),
    coriolis = nothing,
    buoyancy = nothing,
    closure = nothing, 
    momentum_advection = nothing,
    tracer_advection = nothing,
    Δt = 60,
    Nz = 8,
)

rmodel = GordonBell25.baroclinic_instability_model(ReactantState(); kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(); kw...)

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)
@jit Oceananigans.TimeSteppers.update_state!(rmodel)
GordonBell25.compare_states(rmodel, vmodel)

function loop_compute_w!(w, u, v, arch, grid, Nt)
    @trace track_numbers=false for n = 1:Nt
        compute_w!(w, u, v, arch, grid)
    end
end

passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-affine-to-stablehlo{prefer_while_raising=false dump_failed_lockstep=true},canonicalize,arith-raise{stablehlo=true}"

ru, rv, rw = rmodel.velocities
vu, vv, vw = vmodel.velocities
rcompute! = @compile sync=true raise=true compute_w!(rw, ru, rv, rmodel.architecture, rmodel.grid)
rcompute!(rw, ru, rv, rmodel.architecture, rmodel.grid)
compute_w!(vw, vu, vv, vmodel.architecture, vmodel.grid)
GordonBell25.compare_states(rmodel, vmodel)

loop_compute_w!(vw, vu, vv, vmodel.architecture, vmodel.grid, 1)
rloop! = @compile sync=true raise=true loop_compute_w!(rw, ru, rv, rmodel.architecture, rmodel.grid, ConcreteRNumber(1))
rloop!(rw, ru, rv, rmodel.architecture, rmodel.grid, ConcreteRNumber(1))
GordonBell25.compare_states(rmodel, vmodel)

