using PrecompileTools: @setup_workload, @compile_workload
using Oceananigans.Architectures

# Julia v1.11.3 caches bad code: <https://github.com/EnzymeAD/Reactant.jl/issues/614>.
if true  # VERSION != v"1.11.3"
    @setup_workload begin
        @compile_workload begin
            _grid(Architectures.ReactantState())
        end
    end
end
