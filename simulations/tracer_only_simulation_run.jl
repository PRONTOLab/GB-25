using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!, preamble
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

preamble()

Nt = ConcreteRNumber(3)
Oceananigans.defaults.FloatType = Float32

@info "Generating model..."
arch = ReactantState()
model = GordonBell25.tracer_only_model(arch, Nx=32, Ny=32, Nz=32, Δt=1e-2)

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
