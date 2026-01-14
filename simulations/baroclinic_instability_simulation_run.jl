using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: baroclinic_instability_model
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

preamble()

Ninner = ConcreteRNumber(3)
Oceananigans.defaults.FloatType = Float32

@info "Generating model..."
arch = ReactantState()
#arch = Distributed(ReactantState(), partition=Partition(2, 2, 1))
model = baroclinic_instability_model(arch, resolution=8, Δt=60, Nz=10)

GC.gc(true); GC.gc(false); GC.gc(true)

using InteractiveUtils
using KernelAbstractions: @kernel, @index

using Oceananigans: initialize!
using Oceananigans.Utils: launch!, _launch!, configure_kernel, work_layout, mapped_kernel
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: update_state!, time_step!, compute_tendencies!
using Oceananigans.Fields: immersed_boundary_condition

using Oceananigans.ImmersedBoundaries: get_active_cells_map
using Oceananigans.Models: interior_tendency_kernel_parameters
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_tendency_contributions!, compute_hydrostatic_momentum_tendencies!,
                                                        compute_hydrostatic_free_surface_Gc!


using Oceananigans.BuoyancyFormulations
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ
using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∇_dot_qᶜ
using Oceananigans.Biogeochemistry: biogeochemical_transition, biogeochemical_drift_velocity, biogeochemical_auxiliary_fields
using Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₃ⱼ, immersed_∇_dot_qᶜ
using Oceananigans.Advection: div_Uc, U_dot_∇u, U_dot_∇v
using Oceananigans.Forcings: with_advective_forcing
using Oceananigans.TurbulenceClosures: shear_production, buoyancy_flux, dissipation, closure_turbulent_velocity
using Oceananigans.Utils: sum_of_velocities
using KernelAbstractions: @private

import Oceananigans.TurbulenceClosures: hydrostatic_turbulent_kinetic_energy_tendency

function my_compute_tendencies!(model)

    grid = model.grid
    arch = architecture(grid)

    my_compute_hydrostatic_free_surface_tendency_contributions!(model, :xyz; active_cells_map=nothing)

    return nothing
end

function my_compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; active_cells_map=nothing)

    arch = model.architecture
    grid = model.grid

    tracer_index = 1
    tracer_name = :T

    @inbounds c_tendency    = model.timestepper.Gⁿ[tracer_name]
    @inbounds c_advection   = model.advection[tracer_name]
    @inbounds c_forcing     = model.forcing[tracer_name]
    @inbounds c_immersed_bc = immersed_boundary_condition(model.tracers[tracer_name])

    args = tuple(Val(tracer_index),
                    Val(tracer_name),
                    c_advection,
                    model.closure,
                    c_immersed_bc,
                    model.buoyancy,
                    model.biogeochemistry,
                    model.velocities,
                    model.free_surface,
                    model.tracers,
                    model.diffusivity_fields,
                    model.auxiliary_fields,
                    model.clock,
                    c_forcing)

    _my_launch!(arch, grid, kernel_parameters,
            my_compute_hydrostatic_free_surface_Gc!,
            c_tendency,
            grid,
            args)

    return nothing
end

@inline function _my_launch!(arch, grid, workspec, kernel!, first_kernel_arg, other_kernel_args...)

    workgroup = (16, 16)
    worksize  = (48, 24, 10)

    dev  = Oceananigans.Architectures.device(arch)
    loop! = kernel!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(first_kernel_arg, other_kernel_args...)

    return nothing
end

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function my_compute_hydrostatic_free_surface_Gc!(Gc, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = my_hydrostatic_free_surface_tracer_tendency(i, j, k, grid, args...)
end

@inline free_surface_fields(free_surface) = (; η=free_surface.η)
@inline hydrostatic_fields(velocities, free_surface, tracers) =
    merge((u=velocities.u, v=velocities.v, w=velocities.w),
          tracers,
          free_surface_fields(free_surface))

@inline function my_hydrostatic_free_surface_tracer_tendency(i, j, k, grid,
                                                          val_tracer_index::Val{tracer_index},
                                                          val_tracer_name,
                                                          advection,
                                                          closure,
                                                          c_immersed_bc,
                                                          buoyancy,
                                                          biogeochemistry,
                                                          velocities,
                                                          free_surface,
                                                          tracers,
                                                          diffusivities,
                                                          auxiliary_fields,
                                                          clock,
                                                          forcing) where tracer_index

    @inbounds c = tracers[tracer_index]
    model_fields = merge(hydrostatic_fields(velocities, free_surface, tracers),
                         auxiliary_fields,
                         biogeochemical_auxiliary_fields(biogeochemistry))

    biogeochemical_velocities = biogeochemical_drift_velocity(biogeochemistry, val_tracer_name)
    closure_velocities = closure_turbulent_velocity(closure, diffusivities, val_tracer_name)

    total_velocities = sum_of_velocities(velocities, biogeochemical_velocities, closure_velocities)
    total_velocities = with_advective_forcing(forcing, total_velocities)

    return ( - div_Uc(i, j, k, grid, advection, total_velocities, c)
             - ∇_dot_qᶜ(i, j, k, grid, closure, diffusivities, val_tracer_index, c, clock, model_fields, buoyancy)
             - immersed_∇_dot_qᶜ(i, j, k, grid, c, c_immersed_bc, closure, diffusivities, val_tracer_index, clock, model_fields)
             + biogeochemical_transition(i, j, k, grid, biogeochemistry, val_tracer_name, clock, model_fields)
             + forcing(i, j, k, grid, clock, model_fields))
end

@info "Compiling..."
rfirst! = @compile raise=true sync=true my_compute_tendencies!(model)
