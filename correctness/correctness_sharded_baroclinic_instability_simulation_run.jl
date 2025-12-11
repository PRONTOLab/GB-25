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
using KernelAbstractions
using KernelAbstractions: @kernel, @index

using OffsetArrays

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

using Oceananigans.Utils: launch!, _launch!, configure_kernel, work_layout
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
using Oceananigans.Grids: peripheral_node, inactive_cell
using Oceananigans.DistributedComputations: child_architecture







""" Calculate momentum tendencies if momentum is not prescribed."""
function my_compute_hydrostatic_momentum_tendencies!(u, dev, Ny, Nz)

    _my_launch!(dev, my_compute_hydrostatic_free_surface_Gu!, u, Ny, Nz)

    return nothing
end

@inline function _my_launch!(dev, kernel!, first_kernel_arg, other_kernel_args...)

    loop!, worksize = my_configure_kernel(dev, kernel!)

    loop!(first_kernel_arg, other_kernel_args...)

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
    @inbounds Gu[i, j, k] = -my_active_weighted_ℑxyᶠᶜᶜ(i, j, k, Ny, Nz)
end

@inline function my_active_weighted_ℑxyᶠᶜᶜ(i, j, k, Ny, Nz)
    active_nodes = (!((j < 1) | (j > Ny)| (j-1 < 1) | (j-1 > Ny) | (k < 1) | (k > Nz))
                  + !((j+1 < 1) | (j+1 > Ny) | (j < 1) | (j > Ny) | (k < 1) | (k > Nz)))

    mask = active_nodes == 0
    return ifelse(mask, 0.0, 100.0)
end

#@show typeof(rmodel.timestepper.Gⁿ.u.data)
#@show size(rmodel.timestepper.Gⁿ.u.data)

@show vmodel.timestepper.Gⁿ.u
#@show typeof(vmodel.timestepper.Gⁿ.u.data)
#@show size(vmodel.timestepper.Gⁿ.u.data)

U  = zeros(Float64, 128, 128, 32)
vU = OffsetArray(U, -7:120, -7:120, -7:24)
rU = deepcopy(rmodel.timestepper.Gⁿ.u.data) #Reactant.to_rarray(vU)

@jit my_compute_hydrostatic_momentum_tendencies!(rU, ReactantKernelAbstractionsExt.ReactantBackend(), Ny, Nz)
my_compute_hydrostatic_momentum_tendencies!(vU, KernelAbstractions.CPU(), Ny, Nz)

@info "After initialization and update state (should be 0, or at most maybe machine precision, but there's a bug):"
rU  = convert(Array, parent(rU))
vU  = parent(vU)

@show maximum(abs.(rU - vU))
