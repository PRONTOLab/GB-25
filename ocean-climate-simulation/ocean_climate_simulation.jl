using Oceananigans
using Oceananigans.Units

using ClimaOcean
using ClimaOcean.DataWrangling: ECCO4Monthly
using OrthogonalSphericalShellGrids: TripolarGrid

using CFTime
using Dates
using Printf

# See visualize_ocean_climate_simulation.jl for information about how to 
# visualize the results of this run.

# Architecture
use_reactant = get(ENV, "use-reactant", false)
if use_reactant == "true" || use_reactant=="1"
    use_reactant = true
elseif use_reactant == "false" || use_reactant=="0"
    use_reactant = false
end


raise = get(ENV, "raise", false)
if raise == "true" || raise=="1"
    raise = true
elseif raise == "false" || raise=="0"
    raise = false
end
if use_reactant
    arch = ReactantState()
else
    arch = CPU() # change this to use GPU
end
if raise
    Reactant.Compiler.Raise[] = true
end

# Horizontal resolution
resolution = 2 # 1/4 for quarter degree
Nx = convert(Int, 360 / resolution)
Ny = convert(Int, 170 / resolution)

# Vertical resolution
Nz = 20 # eventually we want to increase this to between 100-600

# Time step. This must be decreased as resolution is decreased.
Δt = 20minutes
# Δt = 4minutes # resolution = 1/4
# Δt = 2minutes # resolution = 1/8
# Δt = 1minutes # resolution = 1/16, and so on

# Stop time
stop_time = 10days

# Grid setup
z_faces = exponential_z_faces(; Nz, depth=6000, h=30) # may need changing for very large Nz
underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)
bottom_height = regrid_bathymetry(underlying_grid) # adds Earth bathymetry from ETOPO1
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
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(41))

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

    msg = @sprintf("Time: %s, n: %d, Δt: %s, max|u|: (%.2e, %.2e, %.2e) m s⁻¹, \
                   extrema(T): (%.2f, %.2f) ᵒC, wall time: %s \n",
                   prettytime(sim), iteration(sim), prettytime(sim.Δt),
                   umax..., Tmax, Tmin, prettytime(step_time))

    ClimaOcean.@root @info(msg)

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# Output
if arch isa Distributed
    rank = arch.local_rank
    prefix = "ocean_climate_simulation_rank$rank"
else
    prefix = "ocean_climate_simulation_serial"
end

Nz = size(grid, 3)
outputs = merge(ocean.model.velocities, ocean.model.tracers)
surface_writer = JLD2OutputWriter(ocean.model, outputs,
                                  filename = prefix * "_surface.jld2",
                                  indices = (:, :, Nz),
                                  schedule = TimeInterval(3days),
                                  overwrite_existing = true)

simulation.output_writers[:surface] = surface_writer

# Run the simulation
if get(ENV, "dont-run", true)
    run!(simulation)
end

