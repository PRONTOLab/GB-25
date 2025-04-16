using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false
# Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

# GordonBell25.preamble()

Ninner = ConcreteRNumber(3)
Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
#arch = ReactantState()
#arch = Distributed(ReactantState(), partition=Partition(2, 2, 1))
#model = baroclinic_instability_model(arch, resolution=8, Δt=60, Nz=10)
arch = CPU()
Nx = 128
Ny = 64
Nz = 32
H = 6
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=60 * 10)

for out = 1:100
    GordonBell25.loop!(model, 100)
    @info string("iter: ", iteration(model), ", time: ", time(model))
end

#=
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
=#
