using GordonBell25: baroclinic_instability_model_init
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")
Ninner = ConcreteRNumber(3)

@info "Generating model..."
c_model = baroclinic_instability_model_init(CPU(), Δt=2minutes, resolution=1/4)
r_model = baroclinic_instability_model_init(ReactantState(), Δt=2minutes, resolution=1/4)

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
r_first_time_step! = @compile sync=true raise=true first_time_step!(r_model)
r_loop! = @compile sync=true raise=true loop!(r_model, Ninner)

first_time_step!(c_model)
r_first_time_step!(r_model)

@time loop!(c_model, 10)
@time r_loop!(r_model, ConcreteRNumber(10))

