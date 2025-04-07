using GordonBell25
using KernelAbstractions
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity
using Reactant
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, div_xyᶜᶜᶜ, Δzᶜᶜᶜ, Azᶜᶜᶜ

@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)
    @inbounds U.w[i, j, 1] = 0
    for k in 2:grid.Nz+1
        Δw = - Oceananigans.Operators.flux_div_xyᶜᶜᶜ(i, j, k-1, grid, U.u, U.v) / Azᶜᶜᶜ(i, j, k-1, grid)
        @inbounds U.w[i, j, k] = U.w[i, j, k-1] + Δw
    end
end

function compute_w_from_continuity!(model)
    grid = model.grid
    arch = grid.architecture
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = Oceananigans.Grids.halo_size(grid)
    Tx, Ty, _ = Oceananigans.Grids.topology(grid)

    ii = -Hx+2:Nx+Hx-1
    jj = -Hy+2:Ny+Hy-1

    parameters = Oceananigans.Utils.KernelParameters(ii, jj)
    Oceananigans.Utils.launch!(arch, grid, parameters, _compute_w_from_continuity!, model.velocities, grid)

    return nothing
end

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
GordonBell25.compare_states(rmodel, vmodel)

@jit compute_w_from_continuity!(rmodel)
compute_w_from_continuity!(vmodel)
GordonBell25.compare_states(rmodel, vmodel)

