using PrecompileTools: @setup_workload, @compile_workload

# Julia v1.11.3 caches bad code: <https://github.com/EnzymeAD/Reactant.jl/issues/614>.
# Also disable for v1.10 because of bad caching:
# <https://github.com/PRONTOLab/GB-25/pull/29#issuecomment-2716036657>.
if VERSION >= v"1.11" && VERSION != v"1.11.3"
    @setup_workload begin
        @compile_workload begin
            data_free_ocean_climate_model_init(Architectures.ReactantState();
                                               resolution=5, Nz=4)
        end
        Reactant.clear_oc_cache()
    end
end
