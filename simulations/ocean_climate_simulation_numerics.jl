using GordonBell25: ocean_climate_model_init
using Oceananigans
using Oceananigans.Units
using Printf

# Tunable parameters
FT = Float64

resolution = 2 # degree
Nz = 20 # vertical levels

momentum_advection_order = 5
tracer_advection_order = 5
vorticity_order = nothing

target_dir = "/pscratch/sd/n/nloose/GB-25/simulations/" # change to where you want save stuff

# Generate prefix using parameter values
if resolution isa Rational
    res_str = "$(numerator(resolution))_$(denominator(resolution))"
else
    res_str = "$(Int(resolution))"
end
vort_str = vorticity_order !== nothing ? "_vort$(vorticity_order)" : ""

prefix = (
    "Degree$(res_str)_Nz$(Nz)_"
    * "momadv$(momentum_advection_order)_tradv$(tracer_advection_order)"
    * vort_str * "_$(FT)"
    )

arch = GPU()
zstar_vertical_coordinate = true

Oceananigans.defaults.FloatType = FT

# Spinup with small time step
Δt = 120 # seconds

model = ocean_climate_model_init(arch; resolution, Nz, Δt,
                                 tracer_advection_order,
                                 vorticity_order,
                                 momentum_advection_order,
                                 zstar_vertical_coordinate)

simulation = Simulation(model; Δt, stop_time=10days)

# Callbacks
wall_clock = Ref(time_ns())

function progress(sim)

    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.ocean.model.velocities.u),
                   maximum(abs, sim.model.ocean.model.velocities.v),
                   maximum(abs, sim.model.ocean.model.velocities.w))

    @info msg

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, TimeInterval(1days))

# Diagnostics
u, v, w = model.ocean.model.velocities
e = @at (Center, Center, Center) (u^2 + v^2) / 2
E = Integral(e, dims=(1, 2, 3))

ke_ow = JLD2Writer(
    model.ocean.model,
    (; E),
    filename = "$(prefix)_kinetic_energy.jld2",
    dir = target_dir,
    schedule = AveragedTimeInterval(1days; window=1days, stride=1),
    overwrite_existing = true,
    array_type = Array{FT}
)
simulation.output_writers[:ke] = ke_ow

run!(simulation)

# The real simulation with a larger time step
