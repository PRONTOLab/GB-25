using PrecompileTools: @setup_workload, @compile_workload

# Julia v1.11.3 caches bad code: <https://github.com/EnzymeAD/Reactant.jl/issues/614>.
if VERSION != v"1.11.3"
    @setup_workload begin
        @compile_workload begin
            data_free_ocean_climate_simulation_init(Architectures.ReactantState();
                                                    resolution=3, Nx=120, Ny=60, Nz=4)
        end
        Reactant.clear_oc_cache()
    end
end
