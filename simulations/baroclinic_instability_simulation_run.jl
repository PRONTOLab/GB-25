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

using Oceananigans: initialize!
using Oceananigans.Utils: launch!, _launch!, configure_kernel
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: update_state!, time_step!, compute_tendencies!
using Oceananigans.Fields: immersed_boundary_condition

using Oceananigans.ImmersedBoundaries: get_active_cells_map
using Oceananigans.Models: interior_tendency_kernel_parameters
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_hydrostatic_free_surface_tendency_contributions!, compute_hydrostatic_momentum_tendencies!,
                                                        compute_hydrostatic_free_surface_Gc!

function my_compute_tendencies!(model)

    grid = model.grid
    arch = architecture(grid)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain. The active cells map restricts the computation to the active cells in the
    # interior if the grid is _immersed_ and the `active_cells_map` kwarg is active
    active_cells_map = get_active_cells_map(model.grid, Val(:interior))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    my_compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; active_cells_map)

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
            compute_hydrostatic_free_surface_Gc!,
            c_tendency,
            grid,
            args;
            active_cells_map)

    return nothing
end

@inline function _my_launch!(arch, grid, workspec, kernel!, first_kernel_arg, other_kernel_args...;
                          exclude_periphery = false,
                          reduced_dimensions = (),
                          active_cells_map = nothing)

    location = Oceananigans.location(first_kernel_arg)

    loop!, worksize = configure_kernel(arch, grid, workspec, kernel!;
                                       location,
                                       exclude_periphery,
                                       reduced_dimensions,
                                       active_cells_map)

    # Don't launch kernels with no size
    haswork = true

    if haswork
        loop!(first_kernel_arg, other_kernel_args...)
    end

    return nothing
end

@info "Compiling..."
rfirst! = @compile raise=true sync=true my_compute_tendencies!(model)
