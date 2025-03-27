using GordonBell25: baroclinic_instability_model
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")
Ninner = ConcreteRNumber(3)

@info "Generating model..."
arch = ReactantState()
model = baroclinic_instability_model(arch, resolution=1, Δt=60, Nz=50)

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
Reactant.with_profiler("./") do
    rloop!(model, Ninner)
end
@info "Done!"
