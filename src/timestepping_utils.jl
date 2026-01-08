using Reactant
using Oceananigans
import Oceananigans.TimeSteppers: first_time_step!, time_step!
using Reactant_jll: libReactantExtra

const TRY_COMPILE_FAILED = Ref(false)

function try_compile_code(f)
    try
        f()
    catch e
        @error "Failed to compile" exception=(e, catch_backtrace())
        TRY_COMPILE_FAILED[] = true
        Text("""
        // Failed to compile
        //$e
        """)
    end
end

function first_time_step!(model)
    Δt = model.clock.last_Δt
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    return nothing
end

function time_step!(model)
    Δt = model.clock.last_Δt + 0
    #Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end

using Oceananigans.TimeSteppers:
    update_state!,
    tick!,
    calculate_pressure_correction!,
    correct_velocities_and_cache_previous_tendencies!,
    step_lagrangian_particles!,
    QuasiAdamsBashforth2TimeStepper

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    step_free_surface!,
    local_ab2_step!,
    compute_free_surface_tendency!

using InteractiveUtils


using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_condition!, replace_horizontal_vector_halos!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.Fields: tupled_fill_halo_regions!

import Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
import Oceananigans.TimeSteppers: update_state!

using Oceananigans.Biogeochemistry: update_tendencies!

function myupdate!(model)
    grid = model.grid
    
    @show @which tupled_fill_halo_regions!(prognostic_fields(model), grid, model.clock, fields(model), async=true)
    tupled_fill_halo_regions!(prognostic_fields(model), grid, model.clock, fields(model), async=true)

    # @apply_regionally 
    arch = architecture(grid)

    # Calculate contributions to momentum and tracer tendencies from fluxes and volume terms in the
    # interior of the domain. The active cells map restricts the computation to the active cells in the
    # interior if the grid is _immersed_ and the `active_cells_map` kwarg is active
    active_cells_map = get_active_cells_map(model.grid, Val(:interior))
    kernel_parameters = interior_tendency_kernel_parameters(arch, grid)

    Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_hydrostatic_free_surface_tendency_contributions!(model, kernel_parameters; active_cells_map)
    
    Oceananigans.Models.complete_communication_and_compute_buffer!(model, grid, arch)

    # Calculate contributions to momentum and tracer tendencies from user-prescribed fluxes across the
    # boundaries of the domain
    Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_hydrostatic_boundary_tendency_contributions!(model.timestepper.Gⁿ,
                                                         model.architecture,
                                                         model.velocities,
                                                         model.tracers,
                                                         model.clock,
                                                         fields(model),
                                                         model.closure,
                                                         model.buoyancy)

    update_tendencies!(model.biogeochemistry, model)

end

function loop!(model, Ninner)
    #Δt = model.clock.last_Δt + 0
    @trace track_numbers=false for _ = 1:Ninner

    Δt = model.clock.last_Δt

    #ab2_step!(model, Δt)

    # tick!(model.clock, Δt)

    # calculate_pressure_correction!(model, Δt)
    # correct_velocities_and_cache_previous_tendencies!(model, Δt)

        myupdate!(model)

            # fill_halo_regions!(prognostic_fields(model), model.grid, model.clock, fields(model))
    end
    return nothing
end

function preamble(; rendezvous_warn::Union{Nothing,Int}=nothing, rendezvous_terminate::Union{Nothing,Int}=nothing)
    # If we are in GitHub Actions, make `TMPDIR` be a local directory from which we
    # can upload artifacts at the end.
    if get(ENV, "GITHUB_ACTIONS", "false") == "true"
        Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
        ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "..", "tmp"))
    end

    # Unset environment variables which would cause XLA distributed to hang indefinitely.
    for key in ("no_proxy", "http_proxy", "https_proxy", "NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY")
        delete!(ENV, key)
    end

    if rendezvous_warn isa Int || rendezvous_terminate isa Int
        error("""
              Setting rendezvous timeouts in `preamble` is not supported anymore.
              Use `XLA_FLAGS` instead, e.g.
                  XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80"
              """)
    end
end
