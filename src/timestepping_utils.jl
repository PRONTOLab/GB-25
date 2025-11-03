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

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: compute_barotropic_mode!

using InteractiveUtils
function first_time_step!(model)

    sefs = model.free_surface
    barotropic_velocities = sefs.barotropic_velocities

    
    not_reduced_fields = (barotropic_velocities.U,)
        
    f = barotropic_velocities.U
    
   grid = f.grid
   fill_halo_regions!((f.data,),
     	       (f.boundary_conditions,),
                        Oceananigans.Grids.default_indices(3),
     		 (Oceananigans.Fields.instantiated_location(f),),
     		   grid)


    # ABOVE SHOULD BE MINIMIZED FROM THIS
    @show @which fill_halo_regions!((f, ))
    fill_halo_regions!((f, ))

    ## OLD
    #
    # @show @which fill_halo_regions!((barotropic_velocities.U, barotropic_velocities.V))
    # fill_halo_regions!((barotropic_velocities.U, barotropic_velocities.V))

	# Oceananigans.Models.HydrostaticFreeSurfaceModels.initialize_free_surface!(model.free_surface, model.grid, model.velocities)

#    Δt = model.clock.last_Δt
#    @show @which Oceananigans.TimeSteppers.initialize!(model)
#    Oceananigans.TimeSteppers.initialize!(model)
    return nothing
end

function time_step!(model)
    #Δt = model.clock.last_Δt + 0
    #Oceananigans.TimeSteppers.time_step!(model, Δt)
    return nothing
end

function loop!(model, Ninner)
    #Δt = model.clock.last_Δt + 0
    #@trace track_numbers=false for _ = 1:Ninner
    #    Oceananigans.TimeSteppers.time_step!(model, Δt)
    #end
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
