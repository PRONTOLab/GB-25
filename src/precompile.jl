# Utilities for precompiling
using Oceananigans: fields, prognostic_fields

using Oceananigans.Architectures:
    architecture

using Oceananigans.BoundaryConditions:
    fill_halo_regions!

using Oceananigans.Fields:
    tupled_fill_halo_regions!

using Oceananigans.TimeSteppers:
    ab2_step!,
    correct_velocities_and_cache_previous_tendencies!

using Oceananigans.ImmersedBoundaries:
    get_active_cells_map

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    mask_immersed_model_fields!,
    compute_auxiliaries!,
    compute_hydrostatic_momentum_tendencies!,
    interior_tendency_kernel_parameters,
    compute_hydrostatic_boundary_tendency_contributions!,
    compute_hydrostatic_free_surface_Gc!,
    immersed_boundary_condition,
    complete_communication_and_compute_buffer!,
    compute_tendencies!

#=
# For reference

    * mask_immersed_model_fields!(model, grid)
    * tupled_fill_halo_regions!(prognostic_fields(model), grid, model.clock, fields(model))
    * compute_auxiliaries!(model)
    * fill_halo_regions!(model.diffusivity_fields; only_local_halos=true)
    * compute_tendencies!(model, callbacks)
    * ab2_step!(model, Δt)
    * tupled_fill_halo_regions!(prognostic_fields(model), model.grid, model.clock, fields(model))
    * correct_velocities_and_cache_previous_tendencies!(model, Δt)
=#

function tupled_fill_halo_regions_workload!(model)
    tupled_fill_halo_regions!(prognostic_fields(model), model.grid, model.clock, fields(model))
end

function compute_tendencies_workload!(model)
    compute_tendencies!(model, [])
end

function compute_boundary_tendencies_workload!(model)
    compute_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                         model.architecture,
                                                         model.velocities,
                                                         model.tracers,
                                                         model.clock,
                                                         fields(model),
                                                         model.closure,
                                                         model.buoyancy)
end

function compute_interior_momentum_tendencies_workload!(model)
    grid = model.grid
    arch = architecture(grid)
    active_cells_map = get_active_cells_map(model.grid, Val(:interior))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    #compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; active_cells_map)
    
    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; active_cells_map)
    complete_communication_and_compute_buffer!(model, grid, arch)
end

function compute_interior_tracer_tendencies_workload!(model)
    grid = model.grid
    arch = architecture(grid)
    active_cells_map = get_active_cells_map(model.grid, Val(:interior))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    for (tracer_index, tracer_name) in enumerate(propertynames(model.tracers))

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

        launch!(arch, grid, kernel_parameters,
                compute_hydrostatic_free_surface_Gc!,
                c_tendency,
                grid,
                active_cells_map,
                args;
                active_cells_map)
    end
end

function compute_auxiliaries_workload!(model)
    compute_auxiliaries!(model)
end

function fill_halo_regions_workload!(model)
    fill_halo_regions!(model.diffusivity_fields; only_local_halos=true)
end

function ab2_step_workload!(model, Δt)
    ab2_step!(model, Δt)
end

function correct_velocities_and_cache_previous_tendencies_workload!(model, Δt)
    correct_velocities_and_cache_previous_tendencies!(model, Δt)
end

