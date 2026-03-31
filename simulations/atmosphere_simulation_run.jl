using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: moist_baroclinic_wave_model, set_moist_baroclinic_wave!
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant

preamble()

Ninner = ConcreteRNumber(2)

@info "Generating atmosphere model..."
arch = ReactantState()
model = moist_baroclinic_wave_model(arch; Nλ=48, Nφ=24, Nz=10, Δt=2.0)

@info "Setting initial conditions..."
set_moist_baroclinic_wave!(model)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
rfirst! = @compile raise=true sync=true first_time_step!(model)
rstep!  = @compile raise=true sync=true time_step!(model)
rloop!  = @compile raise=true sync=true loop!(model, Ninner)

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
