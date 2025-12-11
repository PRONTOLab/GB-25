using GordonBell25
using Oceananigans
using Reactant

if !GordonBell25.is_distributed_env_present()
    using MPI
    MPI.Init()
end

throw_error = true
include_halos = true
rtol = sqrt(eps(Float64))
atol = 0

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

Rx, Ry = GordonBell25.factors(Ndev)

rarch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition = Partition(Rx, Ry, 1)
)

rank = Reactant.Distributed.local_rank()

H = 8
Tx = 64 * Rx
Ty = 64 * Ry
Nz = 16

Nx = Tx - 2H
Ny = Ty - 2H

model_kw = (
    halo = (H, H, H),
    Δt = 1e-9,
)

varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel
@assert rmodel.architecture isa Distributed

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)

@info "At the beginning:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@jit Oceananigans.initialize!(rmodel, rmodel.grid)
Oceananigans.initialize!(vmodel, vmodel.grid)





using InteractiveUtils

using KernelAbstractions: @kernel, @index

using Oceananigans.Utils: launch!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_tendency_contributions!, compute_hydrostatic_momentum_tendencies!,
                                                        compute_hydrostatic_free_surface_Gu!, hydrostatic_free_surface_u_velocity_tendency, explicit_barotropic_pressure_x_gradient,
							grid_slope_contribution_x, hydrostatic_fields

using Oceananigans.Coriolis: x_f_cross_U, fᶠᶠᵃ
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: immersed_boundary_condition
using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Operators: ∂xᶠᶜᶜ, ℑxᶠᵃᵃ, ℑyᵃᶜᵃ, ℑxyᶠᶜᵃ, Δx_qᶜᶠᶜ, Δx⁻¹ᶠᶜᶜ, active_weighted_ℑxyᶠᶜᶜ, not_peripheral_node
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ
using Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ








""" Calculate momentum tendencies if momentum is not prescribed."""
function my_compute_hydrostatic_momentum_tendencies!(model; active_cells_map=nothing)

    grid = model.grid
    arch = architecture(grid)

    @show @which ℑxᶠᵃᵃ(1, 1, 1, grid, not_peripheral_node, Center(), Face(), Center())

    launch!(arch, grid, :xyz,
            my_compute_hydrostatic_free_surface_Gu!, model.timestepper.Gⁿ.u, grid; active_cells_map)

    return nothing
end

@kernel function my_compute_hydrostatic_free_surface_Gu!(Gu, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds Gu[i, j, k] = -my_active_weighted_ℑxyᶠᶜᶜ(i, j, k, grid)
end

@inline function my_active_weighted_ℑxyᶠᶜᶜ(i, j, k, grid)
    active_nodes = (not_peripheral_node(i-1, j, k, grid, Center(), Face(), Center())
                  + not_peripheral_node(i,   j, k, grid, Center(), Face(), Center())
                  + not_peripheral_node(i-1, j+1, k, grid, Center(), Face(), Center()))

    mask = active_nodes == 0
    return ifelse(mask, zero(grid), 100.0)
end

@jit my_compute_hydrostatic_momentum_tendencies!(rmodel)
my_compute_hydrostatic_momentum_tendencies!(vmodel)

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)





#=

GordonBell25.sync_states!(rmodel, vmodel)
rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
@showtime rfirst!(rmodel)
@showtime GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)

@info "Warm up:"
@showtime rstep!(rmodel)
@showtime rstep!(rmodel)
@showtime GordonBell25.time_step!(vmodel)
@showtime GordonBell25.time_step!(vmodel)

Nt = 10
@info "Time step with Reactant:"
for _ in 1:Nt
    @showtime rstep!(rmodel)
end

@info "Time step vanilla:"
for _ in 1:Nt
    @showtime GordonBell25.time_step!(vmodel)
end

@info "After $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)
@jit Oceananigans.TimeSteppers.update_state!(rmodel)

@info "After syncing and updating state again:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@showtime rloop!(rmodel, rNt)
@showtime GordonBell25.loop!(vmodel, Nt)

@info "After a loop of $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
=#