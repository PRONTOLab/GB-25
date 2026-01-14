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
                              advective_tracer_flux_z, bias, _biased_interpolate_zᵃᵃᶠ, outside_biased_halo_zᶠ,
                              biased_interpolate_zᵃᵃᶠ, _____biased_interpolate_zᵃᵃᶠ

import Oceananigans.TurbulenceClosures: hydrostatic_turbulent_kinetic_energy_tendency
using Oceananigans.Grids: topology, required_halo_size_z

function my_compute_tendencies!(c_tendency, dev, Nz)

    _my_launch!(dev,
            my_compute_hydrostatic_free_surface_Gc!,
            c_tendency,
            Nz)

    return nothing
end

@inline function _my_launch!(dev, kernel!, first_kernel_arg, other_kernel_args...)

    workgroup = (16, 16)
    worksize  = (48, 24, 10)

    loop! = kernel!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(first_kernel_arg, other_kernel_args...)

    return nothing
end

""" Calculate the right-hand-side of the tracer advection-diffusion equation. """
@kernel function my_compute_hydrostatic_free_surface_Gc!(Gc, Nz)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = ifelse(my_outside_biased_halo_zᶠ(k, Nz), 100.0, 3.0)
end

@inline my_outside_biased_halo_zᶠ(i, N) = (i >= 3 + 1) & (i <= N + 1 - (3 - 1)) &  # Left bias
                                          (i >= 3)     & (i <= N + 1 - 3)          # Right bias

@info "Compiling..."

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

dev = ReactantKernelAbstractionsExt.ReactantBackend()
Nz  = 10

rfirst! = @compile raise=true sync=true my_compute_tendencies!(model.timestepper.Gⁿ.T, dev, Nz)

#my_compute_tendencies!(model)
