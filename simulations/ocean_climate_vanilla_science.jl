using GordonBell25: ocean_climate_model_init
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie

# Build the ocean climate model on CPU at 1/6 degree
# Data files are downloaded automatically if not present locally.
arch = CPU()
resolution = 1/6

@info "Building model at $(resolution)° resolution..."
coupled_model = ocean_climate_model_init(arch; resolution)

# Create a simulation
Δt = 5minutes
stop_time = 30days
simulation = Simulation(coupled_model; Δt, stop_time)

# Progress callback
wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    Tmax = maximum(interior(T))
    Tmin = minimum(interior(T))
    umax = (maximum(abs, interior(u)), maximum(abs, interior(v)), maximum(abs, interior(w)))
    elapsed = 1e-9 * (time_ns() - wall_time[])

    @info @sprintf("Time: %s, iter: %d, Δt: %s, max|u|: (%.2e, %.2e, %.2e) m/s, T: (%.2f, %.2f) °C, wall: %s",
                   prettytime(sim), iteration(sim), prettytime(sim.Δt),
                   umax..., Tmin, Tmax, prettytime(elapsed))
    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# Output writers — save surface fields every 12 hours
ocean = coupled_model.ocean
Nz = size(ocean.model.grid, 3)
outputs = merge(ocean.model.velocities, ocean.model.tracers)

simulation.output_writers[:surface] = JLD2OutputWriter(ocean.model, outputs,
    filename = "ocean_climate_surface.jld2",
    indices = (:, :, Nz),
    schedule = TimeInterval(12hours),
    overwrite_existing = true)

# Run
@info "Running simulation for $stop_time..."
run!(simulation)
@info "Simulation complete!"

# ============================================================
# Visualization
# ============================================================

filepath = simulation.output_writers[:surface].filepath

T_ts = FieldTimeSeries(filepath, "T")
u_ts = FieldTimeSeries(filepath, "u")

times = T_ts.times
Nt = length(times)

# Plot snapshots of surface temperature
fig = Figure(size = (1200, 800))

# Pick 4 evenly-spaced snapshots
snapshot_indices = round.(Int, range(1, Nt, length=4))

for (i, n) in enumerate(snapshot_indices)
    row = (i - 1) ÷ 2 + 1
    col = (i - 1) % 2 + 1

    ax = Axis(fig[row, col],
              title = "SST at t = $(prettytime(times[n]))",
              xlabel = "Longitude",
              ylabel = "Latitude")

    Tn = interior(T_ts[n], :, :, 1)
    λ = λnodes(T_ts[n])
    φ = φnodes(T_ts[n])

    hm = heatmap!(ax, λ, φ, Tn, colormap = :thermal)
    Colorbar(fig[row, col][1, 2], hm, label = "T (°C)")
end

save("ocean_climate_surface_temperature.png", fig)
@info "Saved ocean_climate_surface_temperature.png"

# Plot surface speed
fig2 = Figure(size = (1200, 800))

for (i, n) in enumerate(snapshot_indices)
    row = (i - 1) ÷ 2 + 1
    col = (i - 1) % 2 + 1

    ax = Axis(fig2[row, col],
              title = "Surface speed at t = $(prettytime(times[n]))",
              xlabel = "Longitude",
              ylabel = "Latitude")

    un = interior(u_ts[n], :, :, 1)
    speed = abs.(un)

    λ = λnodes(u_ts[n])
    φ = φnodes(u_ts[n])

    hm = heatmap!(ax, λ, φ, speed, colormap = :speed, colorrange = (0, 0.5))
    Colorbar(fig2[row, col][1, 2], hm, label = "|u| (m/s)")
end

save("ocean_climate_surface_speed.png", fig2)
@info "Saved ocean_climate_surface_speed.png"
