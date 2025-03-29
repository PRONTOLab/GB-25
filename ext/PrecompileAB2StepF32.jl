module PrecompileAB2StepF32

using GordonBell25
using Reactant
using Oceananigans
using PrecompileTools: @setup_workload, @compile_workload

#=
using Oceananigans: fields, prognostic_fields

using Oceananigans.BoundaryConditions:
    fill_halo_regions!

using Oceananigans.Fields:
    tupled_fill_halo_regions!

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    mask_immersed_model_fields!,
    compute_auxiliaries!,
    compute_tendencies!

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

@setup_workload begin
    FT = Float32
    Oceananigans.defaults.FloatType = FT
    Nx, Ny, Nz = 64, 32, 4
    arch = Oceananigans.Architectures.ReactantState()

    @compile_workload begin
        Δt = FT(60)
        model = GordonBell25.baroclinic_instability_model(arch; resolution=4, Δt, Nz=4, grid=:simple_lat_lon)
        compiled! = @compile GordonBell25.ab2_step_workload!(model, Δt)
    end

    # Reset float type
    Oceananigans.defaults.FloatType = Float64
    Reactant.clear_oc_cache()
end

end # module

