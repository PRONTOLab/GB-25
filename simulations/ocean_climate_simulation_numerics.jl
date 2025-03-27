using GordonBell25: ocean_climate_model_init
using Oceananigans
using Oceananigans.Units
using Printf

# Generate prefix using parameter values
if resolution isa Rational
    res_str = "$(numerator(resolution))_$(denominator(resolution))"
else
    res_str = "$(Int(resolution))"
end

if arch == GPU()
    arch_str = "GPU"
elseif arch == CPU()
    arch_str = "CPU"
elseif arch == ReactantState()
    arch_str = "Reactant"
end
    
vort_str = vorticity_order !== nothing ? "_vort$(vorticity_order)" : ""

prefix = (
    "Degree$(res_str)_Nz$(Nz)_"
    * "momadv$(momentum_advection_order)_tradv$(tracer_advection_order)"
    * vort_str * "_$(FT)_$(arch_str)"
    )

# Some more parameters that we will NOT change for the numerics tests
zstar_vertical_coordinate = true
include_passive_tracers = true

# Set precision
Oceananigans.defaults.FloatType = FT

# Spinup with small time step
model = ocean_climate_model_init(arch; 
    resolution=resolution, 
    Nz=Nz, 
    Δt=Δt₁,
    tracer_advection_order=tracer_advection_order,
    vorticity_order=vorticity_order,
    momentum_advection_order=momentum_advection_order,
    zstar_vertical_coordinate=zstar_vertical_coordinate,
    include_passive_tracers=include_passive_tracers
)

# Run spin-up of 30 days with small time step
simulation = Simulation(model; Δt=Δt₁, stop_time=30days)

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

add_callback!(simulation, progress, TimeInterval(10days))

# Integral diagnostics
u, v, w = model.ocean.model.velocities
e = @at (Center, Center, Center) (u^2 + v^2) / 2
kinetic_energy_integral = Integral(e, dims=(1, 2, 3))

c_surface, c_bottom = model.ocean.model.tracers.C_surface, model.ocean.model.tracers.C_bottom 

surface_tracer_integral = Integral(c_surface, dims=(1, 2, 3))
bottom_tracer_integral = Integral(c_bottom, dims=(1, 2, 3))
surface_tracer_variance_integral = Integral(c_surface^2, dims=(1, 2, 3))
bottom_tracer_variance_integral = Integral(c_bottom^2, dims=(1, 2, 3))

integral_ow = JLD2Writer(
    model.ocean.model,
    (; kinetic_energy_integral, surface_tracer_integral, bottom_tracer_integral, surface_tracer_variance_integral, bottom_tracer_variance_integral),
    filename = "$(prefix)_integrals",
    dir = target_dir,
    schedule = AveragedTimeInterval(1days; window=1days, stride=1),
    overwrite_existing = true,
    array_type = Array{FT}
)
simulation.output_writers[:integral] = integral_ow

# Surface field diagnostics
outputs = merge(model.ocean.model.tracers, model.ocean.model.velocities)

surface_field_ow = JLD2Writer(
    model.ocean.model, 
    outputs,
    filename ="$(prefix)_surface_fields",
    dir = target_dir,
    indices = (:, :, Nz),
    schedule = AveragedTimeInterval(1days; window=1days, stride=1),
    overwrite_existing = true,
    array_type = Array{FT}
)
simulation.output_writers[:surface] = surface_field_ow

# Vertically integrated tracer diagnostics
surface_tracer_vertical_integral = Integral(c_surface, dims=(3))
bottom_tracer_vertical_integral = Integral(c_bottom, dims=(3))

vertical_integral_ow = JLD2Writer(
    model.ocean.model, 
    (; surface_tracer_vertical_integral, bottom_tracer_vertical_integral),
    filename ="$(prefix)_vertically_integrated_tracers",
    dir = target_dir,
    indices = (:, :, Nz),
    schedule = AveragedTimeInterval(1days; window=1days, stride=1),
    overwrite_existing = true,
    array_type = Array{FT}
)
simulation.output_writers[:vertical_integral] = vertical_integral_ow

# Run the simulation
run!(simulation)

# Proceed simulation with a larger time step
simulation.stop_time = 730days
simulation.Δt = Δt₂
run!(simulation)
