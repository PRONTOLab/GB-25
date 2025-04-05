using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: baroclinic_instability_model
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant

Reactant.Compiler.WHILE_CONCAT[] = true
Reactant.Compiler.DUS_TO_CONCAT[] = true
# Reactant.Compiler.SROA_ATTRIBUTOR[] = false
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

preamble()

Ninner = ConcreteRNumber(1000)
Oceananigans.defaults.FloatType = Float32

@info "Generating model..."
arch = ReactantState()
#arch = Distributed(ReactantState(), partition=Partition(2, 2, 1))
model = baroclinic_instability_model(arch, Δt=60, Nx=256, Ny=256, Nz=128)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true sync=true first_time_step!(model)
rstep! = @compile raise=true sync=true time_step!(model)
rloop! = @compile raise=true sync=true loop!(model, Ninner)

@info "Running..."
Reactant.with_profiler("./rfirst") do
    rfirst!(model)
end
Reactant.with_profiler("./rstep") do
    rstep!(model)
end
Reactant.with_profiler("./rloop0") do
    rloop!(model, Ninner)
end
Reactant.with_profiler("./rloop") do
    rloop!(model, Ninner)
end
@info "Done!"
