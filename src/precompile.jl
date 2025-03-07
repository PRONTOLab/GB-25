using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        _grid(Architectures.ReactantState())
    end
    Reactant.clear_oc_cache()
end
