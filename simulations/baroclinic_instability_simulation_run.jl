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

using Oceananigans.Advection: _advective_tracer_flux_x, _advective_tracer_flux_y, _advective_tracer_flux_z,
                              advective_tracer_flux_z, bias, _biased_interpolate_zᵃᵃᶠ

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

    #@show @which _biased_interpolate_zᵃᵃᶠ(1, 1, 1, grid, c_advection, bias(model.velocities.w[1,1,1]), model.tracers[tracer_index])

    #@show bias(model.velocities.w[1,1,1])

    _my_launch!(arch, grid, kernel_parameters,
            my_compute_hydrostatic_free_surface_Gc!,
            c_tendency,
            grid,
            c_advection,
            model.tracers[tracer_name])

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
@kernel function my_compute_hydrostatic_free_surface_Gc!(Gc, grid, scheme, c)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = _biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, Oceananigans.Advection.RightBias(), c)
end

@info "Compiling..."
rfirst! = @compile raise=true sync=true my_compute_tendencies!(model)

#my_compute_tendencies!(model)
