using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState())
Ninner = ConcreteRNumber(2)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true sync=true first_time_step!(model)
rstep! = @compile raise=true sync=true time_step!(model)
rloop! = @compile raise=true sync=true loop!(model, Ninner)

@info "Running..."
Reactant.with_profiler("./") do
    rfirst!(model)
end
Reactant.with_profiler("./") do
    rstep!(model)
end
Reactant.with_profiler("./";
    advanced_config=Dict{String,String}(
        "gpu_pm_sample_counters" => join([
            "sm__cycles_active.avg",
            "sm__inst_executed_realtime.avg.per_cycle_active",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__t_sector_hit_rate.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__inst_executed.sum",
        ], ","),
    )) do
    rloop!(model, Ninner)
end
@info "Done!"
