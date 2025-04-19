using Dates
using GordonBell25
using Reactant
using KernelAbstractions: @kernel, @index
using Oceananigans
using Printf

# This must be called before `GordonBell25.initialize`!
GordonBell25.preamble(; rendezvous_warn=20, rendezvous_terminate=40)
@show Ndev = length(Reactant.devices())
local_arch = Oceananigans.ReactantState()
# local_arch = CPU()

if Ndev == 1
    Rx = Ry = 1
    rank = 0
    arch = local_arch
else
    Rx, Ry = GordonBell25.factors(Ndev)
    arch = Oceananigans.Distributed(
        local_arch,
        partition = Partition(Rx, Ry, 1)
    )

    rank = arch.local_rank
end

@show arch

H = 8
Nx = 48 * Rx
Ny = 24 * Ry
Nz = 4

@info "[$rank] Generating model..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@show model

#=
free_surface_func!(model) = Oceananigans.Models.HydrostaticFreeSurfaceModels.step_free_surface!(
    model.free_surface,
    model,
    model.timestepper,
    model.clock.last_Δt)
=#

function free_surface_func!(model)
    substepping = model.free_surface.substepping
    Ns = Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces.calculate_substeps(substepping, model.clock.last_Δt)
    δ, w = Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces.calculate_adaptive_settings(substepping, Ns)
    Δτ = δ * model.clock.last_Δt
    #Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces.iterate_split_explicit!(
    simple_iterate_split_explicit!(
        model.free_surface,
        model.free_surface.η.grid,
        model.timestepper.Gⁿ.U,
        model.timestepper.Gⁿ.V,
        Δτ,
        w,
        Val(Ns)
    )

    return nothing
end
    
# using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
#     _split_explicit_free_surface!,
#     _split_explicit_barotropic_velocity!

using Oceananigans.Operators:
    δxTᶜᵃᵃ,
    δyTᵃᶜᵃ,
    ∂xTᶠᶜᶠ,
    ∂yTᶜᶠᶠ,
    Δy_qᶠᶜᶠ,
    Δx_qᶜᶠᶠ,
    Azᶜᶜᶠ

@kernel function _split_explicit_free_surface!(grid, Δτ, η, U, V)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1
        
    @inbounds  η[i, j, k_top] -= Δτ * (δxTᶜᵃᵃ(i, j, grid.Nz, grid, Δy_qᶠᶜᶠ, U) +
                                       δyTᵃᶜᵃ(i, j, grid.Nz, grid, Δx_qᶜᶠᶠ, V)) / Azᶜᶜᶠ(i, j, k_top, grid)
end

@kernel function _split_explicit_barotropic_velocity!(averaging_weight, grid, Δτ, η, U, V, η̅, U̅, V̅, Gᵁ, Gⱽ, g) 
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    # Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, η)
    # Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, η)
    
    @inbounds begin
        # ∂τ(U) = - ∇η + G
        # U[i, j, 1] +=  Δτ * (- g * Hᶠᶜ * ∂xTᶠᶜᶠ(i, j, k_top, grid, η) + Gᵁ[i, j, 1])
        # V[i, j, 1] +=  Δτ * (- g * Hᶜᶠ * ∂yTᶜᶠᶠ(i, j, k_top, grid, η) + Gⱽ[i, j, 1])

        U[i, j, 1] += Δτ * (Gᵁ[i, j, 1] - ∂xTᶠᶜᶠ(i, j, k_top, grid, η))
        V[i, j, 1] += Δτ * (Gⱽ[i, j, 1] - ∂yTᶜᶠᶠ(i, j, k_top, grid, η))
        # U[i, j, 1] += Δτ * Gᵁ[i, j, 1]
        # V[i, j, 1] += Δτ * Gⱽ[i, j, 1] 
                            
        # time-averaging
        η̅[i, j, k_top] += averaging_weight * η[i, j, k_top]
        U̅[i, j, 1] += averaging_weight * U[i, j, 1]
        V̅[i, j, 1] += averaging_weight * V[i, j, 1]
    end
end

function simple_iterate_split_explicit!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, weights, ::Val{Nsubsteps}) where Nsubsteps
    arch = grid.architecture
    η           = free_surface.η
    state       = free_surface.filtered_state
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    # unpack state quantities, parameters and forcing terms
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η, state.U, state.V

    free_surface_kernel!, _ = Oceananigans.Utils.configure_kernel(arch, grid, parameters, _split_explicit_free_surface!)
    barotropic_velocity_kernel!, _ = Oceananigans.Utils.configure_kernel(arch, grid, parameters, _split_explicit_barotropic_velocity!)

    η_args = (grid, Δτᴮ, η, U, V)
    U_args = (grid, Δτᴮ, η, U, V, η̅, U̅, V̅, GUⁿ, GVⁿ, g) 

    GC.@preserve η_args U_args begin
        # We need to perform ~50 time-steps which means
        # launching ~100 very small kernels: we are limited by
        # latency of argument conversion to GPU-compatible values.
        # To alleviate this penalty we convert first and then we substep!
        converted_η_args = Oceananigans.Architectures.convert_to_device(arch, η_args)
        converted_U_args = Oceananigans.Architectures.convert_to_device(arch, U_args)

        for substep in 1:Nsubsteps
            Base.@_inline_meta
            averaging_weight = weights[substep]
            free_surface_kernel!(converted_η_args...)
            barotropic_velocity_kernel!(averaging_weight, converted_U_args...)
        end
    end

    return nothing
end

function myloop!(model, Nt)
    @trace track_numbers=false for _ = 1:Nt
        free_surface_func!(model)
    end
    return nothing
end

if local_arch isa CPU
    free_surface_func!(model)
    error("done")
end

name = "simple_iterate_free_surface"
xla_code = @code_xla raise=true free_surface_func!(model)
open("sharded_$(name)_$Ndev.xla.txt", "w") do io
    print(io, xla_code)
end

xla_loop_code = @code_xla raise=true myloop!(model, ConcreteRNumber(3))
open("sharded_$(name)_loop_$Ndev.xla.txt", "w") do io
    print(io, xla_loop_code)
end
