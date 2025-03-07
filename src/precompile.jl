using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        data_free_ocean_climate_simulation_init()
    end
    Reactant.clear_oc_cache()
end
