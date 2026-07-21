# Utilities for precompiling
using Oceananigans: fields, prognostic_fields

using Oceananigans.Architectures:
    architecture

using Oceananigans.BoundaryConditions:
    fill_halo_regions!

using Oceananigans.TimeSteppers:
    ab2_step!,
    cache_previous_tendencies!

using Oceananigans.Grids:
    get_active_cells_map

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    mask_immersed_model_fields!,
    compute_closure_fields!,
    compute_hydrostatic_momentum_tendencies!,
    compute_momentum_tendencies!,
    compute_tracer_tendencies!,
    interior_tendency_kernel_parameters,
    compute_hydrostatic_free_surface_Gc!,
    immersed_boundary_condition

#=
# For reference (update_state! + ab2_step! flow)

    * mask_immersed_model_fields!(model)
    * fill_halo_regions!((u, v), model.clock, fields(model))
    * fill_halo_regions!(tracers, model.clock, fields(model))
    * compute_closure_fields!(model.closure_fields, model.closure, model, ...)
    * fill_halo_regions!(model.closure_fields; only_local_halos=true)
    * compute_momentum_tendencies!(model, callbacks)
    * ab2_step!(model, Δt)
    * fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))
    * cache_previous_tendencies!(model)
=#

function fill_halo_regions_prognostic_workload!(model)
    fill_halo_regions!(prognostic_fields(model), model.clock, fields(model))
end

function compute_tendencies_workload!(model)
    compute_momentum_tendencies!(model, [])
    compute_tracer_tendencies!(model)
end

function compute_interior_momentum_tendencies_workload!(model)
    grid = model.grid
    arch = architecture(grid)
    active_cells_map = get_active_cells_map(model.grid, Val(:interior))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    #compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; active_cells_map)
    
    compute_hydrostatic_momentum_tendencies!(model, model.velocities, kernel_parameters; active_cells_map)
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
                     model.transport_velocities,
                     model.free_surface,
                     model.tracers,
                     model.closure_fields,
                     model.auxiliary_fields,
                     model.clock,
                     c_forcing)

        launch!(arch, grid, kernel_parameters,
                compute_hydrostatic_free_surface_Gc!,
                c_tendency,
                grid,
                args;
                active_cells_map)
    end
end

function compute_auxiliaries_workload!(model)
    compute_closure_fields!(model.closure_fields, model.closure, model)
end

function fill_halo_regions_workload!(model)
    fill_halo_regions!(model.closure_fields; only_local_halos=true)
end

function ab2_step_workload!(model, Δt)
    ab2_step!(model, Δt)
end

function cache_previous_tendencies_workload!(model)
    cache_previous_tendencies!(model)
end

