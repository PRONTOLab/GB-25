using GordonBell25: ocean_climate_model_init
using Oceananigans

FT = Float64
Oceananigans.defaults.FloatType = FT

arch = CPU()
resolution = 2 # degree
Nz = 20 # vertical levels
Δt = 60 # seconds, note this must be provided at initialization for Reactant
momentum_advection_order = 5
tracer_advection_order = 5

# Set this to eg vorticity_order=7, which additionally ignores `momentum_advection_order`
vorticity_order = nothing
model = ocean_climate_model_init(arch; resolution, Nz, Δt,
                                 tracer_advection_order,
                                 vorticity_order,
                                 momentum_advection_order)

simulation = Simulation(model; Δt, stop_iteration=10)

# Add diagnostics and callbacks here

run!(simulation)

