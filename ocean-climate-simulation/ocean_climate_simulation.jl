using Oceananigans
using Oceananigans.Units

using ClimaOcean
using ClimaOcean.DataWrangling: ECCO4Monthly
using OrthogonalSphericalShellGrids: TripolarGrid

using CFTime
using Dates
using Printf

# Turn this off if you can't make plots with GLMakie
make_visualization = true

# Architecture
arch = CPU()

# Horizontal resolution
resolution = 2 # 1/4 for quarter degree
Nx = convert(Int, 360 / resolution)
Ny = convert(Int, 170 / resolution)

# Vertical resolution
Nz = 20 #100

# Time step. This must be decreased as resolution is decreased.
Δt = 20minutes
# Δt = 4minutes # resolution = 1/4
# Δt = 2minutes # resolution = 1/8
# Δt = 1minutes # resolution = 1/16, and so on

# Stop time
stop_time = 10days

# Grid setup
z_faces = exponential_z_faces(; Nz, depth=6000, h=30)
underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)
bottom_height = regrid_bathymetry(underlying_grid)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

# Polar restoring setup
dates = DateTimeProlepticGregorian(1993, 1, 1) : Month(1) : DateTimeProlepticGregorian(1993, 12, 1)
temperature = ECCOMetadata(:temperature, dates, ECCO4Monthly())
salinity = ECCOMetadata(:salinity, dates, ECCO4Monthly())

restoring_rate  = 1/7days
mask = LinearlyTaperedPolarMask(southern=(-80, -70), northern=(70, 90))
FT = ECCORestoring(temperature, grid; mask, rate=restoring_rate)
FS = ECCORestoring(salinity, grid; mask, rate=restoring_rate)

# Ocean simulation with defaults from ClimaOcean
ocean = ocean_simulation(grid; forcing=(T=FT, S=FT))

# Initial ocean state from ECCO state estimate
set!(ocean.model, T=ECCOMetadata(:temperature; dates=first(dates)),
                  S=ECCOMetadata(:salinity; dates=first(dates)))

# Atmospheric model
radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(20))

# Coupled model and simulation
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation) 
simulation = Simulation(coupled_model; Δt, stop_time)

# Utility for printing progress to the terminal
wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    Tmax = maximum(interior(T))
    Tmin = minimum(interior(T))
    umax = (maximum(abs, interior(u)), maximum(abs, interior(v)), maximum(abs, interior(w)))
    step_time = 1e-9 * (time_ns() - wall_time[])

    @info @sprintf("Time: %s, n: %d, Δt: %s, max|u|: (%.2e, %.2e, %.2e) m s⁻¹, \
                   extrema(T): (%.2f, %.2f) ᵒC, wall time: %s \n",
                   prettytime(sim), iteration(sim), prettytime(sim.Δt),
                   umax..., Tmax, Tmin, prettytime(step_time))

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# Output
Nz = size(grid, 3)
prefix = "ocean_climate_simulation"
outputs = merge(model.velocities, model.tracers)
surface_writer = JLD2OutputWriter(ocean.model, outputs,
                                  filename = prefix * "_surface.jld2",
                                  indices = (:, :, Nz),
                                  schedule = TimeInterval(3days),
                                  overwrite_existing=true)

simulation.output_writers[:surface] = surface_writer

# Run the simulation
run!(simulation)

# Optional visualization of the final state
if make_visualization
    # A simple visualization
    using GLMakie

    fig = Figure(size=(600, 700))
    axT = Axis(fig[1, 1])
    axu = Axis(fig[2, 1])

    T = ocean.model.tracers.T
    u = ocean.model.velocities.u
    Nz = size(grid, 3)
    heatmap!(axT, interior(T, :, :, Nz))
    heatmap!(axu, interior(u, :, :, Nz))
    display(fig)
end

