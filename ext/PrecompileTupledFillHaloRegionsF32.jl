module PrecompileTupledFillHaloRegionsF32

using GordonBell25
using Reactant
using Oceananigans
using PrecompileTools: @setup_workload, @compile_workload

#=
using Oceananigans: fields, prognostic_fields

using Oceananigans.BoundaryConditions:
    fill_halo_regions!

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    mask_immersed_model_fields!,
    compute_closure_fields!,
    compute_momentum_tendencies!,
    compute_tracer_tendencies!

# For reference

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

@setup_workload begin
    FT = Float32
    Oceananigans.defaults.FloatType = FT
    Nx, Ny, Nz = 64, 32, 4
    arch = Oceananigans.Architectures.ReactantState()

    @compile_workload begin
        model = GordonBell25.baroclinic_instability_model(arch; resolution=4, Δt=60, Nz=4, grid_type=:simple_lat_lon)
        compiled! = @compile GordonBell25.fill_halo_regions_prognostic_workload!(model)
    end

    # Reset float type
    Oceananigans.defaults.FloatType = Float64
    Reactant.clear_oc_cache()
end

end # module

