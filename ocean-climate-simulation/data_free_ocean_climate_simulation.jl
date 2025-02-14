using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures
using Reactant

using ClimaOcean
using ClimaOcean.DataWrangling: ECCO4Monthly
using OrthogonalSphericalShellGrids: TripolarGrid

using CFTime
using Dates
using Printf

# See visualize_ocean_climate_simulation.jl for information about how to 
# visualize the results of this run.

# Architecture
arch = CPU() # change this to use GPU
use_reactant = get(ENV, "use-reactant", false)
if use_reactant == "true" || use_reactant=="1"
    use_reactant = true
    arch = Oceananigans.Architectures.ReactantState()
elseif use_reactant == "false" || use_reactant=="0"
    use_reactant = false
end

raise = get(ENV, "raise", false)
if raise == "true" || raise=="1"
    raise = true
    Reactant.Compiler.Raise[] = true
elseif raise == "false" || raise=="0"
    raise = false
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

φ₁ = φ₂ = 55
λ₁ = 70
λ₂ = λ₁ + 180
dφ = 10
island₁(φ, λ) = exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
island₂(φ, λ) = exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
zb = z_faces[1]
h = -zb + 1000
gaussian_islands(λ, φ) = zb + h * (island₁(λ, φ) + island₂(λ, φ))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands))

ocean = ocean_simulation(grid)

# Simple initial condition for producing pretty pictures
φ₀ = 40
dφ = 5
dTdz = 1e-3
dSdz = - 5e-3
smooth_step(φ) = (1 - tanh((abs(φ) - φ₀) / dφ)) / 2
Tᵢ(λ, φ, z) = (30 + dTdz * z) * smooth_step(φ) + rand()
Sᵢ(λ, φ, z) = dSdz * z + rand()
set!(ocean.model, T=Tᵢ, S=Sᵢ)

# Set up an atmosphere
atmos_times = range(0, 1day, length=24)
atmosphere = PrescribedAtmosphere(grid, atmos_times)

zonal_wind(λ, φ) = 4 * sind(2φ)^2 - 2 * exp(-(abs(φ) - 12)^2 / 72)
sunlight(λ, φ) = -200 - 600 * cosd(φ)^2

Ta = CenterField(atmos_grid)
ua = CenterField(atmos_grid)
Qsw = CenterField(atmos_grid)

Tatm(λ, φ, z=0) = 30 * cosd(φ)
set!(Ta, Tatm)
set!(ua, zonal_wind)
set!(Qsw, sunlight)

parent(atmosphere.tracers.T) .= parent(Ta) .+ 273.15
parent(atmosphere.velocities.u) .= parent(ua)
parent(atmosphere.tracers.q) .= 0
parent(atmosphere.downwelling_radiation.shortwave) .= parent(Qsw)

# Atmospheric model
radiation  = Radiation(arch)

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

