using MPI
MPI.Init()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids
using Oceananigans.Architectures
using Reactant

using ClimaOcean
using ClimaOcean.DataWrangling: ECCO4Monthly
using ClimaOcean.OceanSeaIceModels.InterfaceComputations: FixedIterations, ComponentInterfaces
using OrthogonalSphericalShellGrids: TripolarGrid

using Dates
using Printf


ranks = MPI.Comm_size(MPI.COMM_WORLD)
arch = Distributed(GPU(), partition = Partition(y = ranks))

# Grid size
Nx = 4320
Ny = 2160
Nz = 100

# Grid setup
r_faces = exponential_z_faces(; Nz, depth=6000, h=30) # may need changing for very large Nz
z_faces = MutableVerticalDiscretization(r_faces)
underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)
bottom_height = regrid_bathymetry(underlying_grid) # adds Earth bathymetry from ETOPO1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

# Polar restoring setup
dates = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 12, 1)
temperature = ECCOMetadata(:temperature, dates, ECCO4Monthly())
salinity = ECCOMetadata(:salinity, dates, ECCO4Monthly())

restoring_rate  = 1/7days
mask = LinearlyTaperedPolarMask(southern=(-80, -70), northern=(70, 90), z=(-10000, 0))
FT = ECCORestoring(temperature, grid; mask, rate=restoring_rate)
FS = ECCORestoring(salinity, grid; mask, rate=restoring_rate)

@inline function damp_u_velocity(i, j, k, grid, clock, fields, mask)
    λ, φ, z = node(i, j, k, grid, Face(), Center(), Center())
    return - mask(φ, z) * fields.u[i, j, k]
end

@inline function damp_v_velocity(i, j, k, grid, clock, fields, mask)
    λ, φ, z = node(i, j, k, grid, Center(), Face(), Center())
    return - mask(φ, z) * fields.v[i, j, k]
end

Fu = Forcing(damp_u_velocity, discrete_form=true, parameters=mask)
Fv = Forcing(damp_v_velocity, discrete_form=true, parameters=mask)

# Improve advection schemes
momentum_advection = WENOVectorInvariant()
tracer_advection = WENO(order=7)

# Ocean simulation with defaults from ClimaOcean
ocean = ocean_simulation(grid; 
                         forcing=(T=FT, S=FT, u=Fu, v=Fv), 
                         momentum_advection, 
                         tracer_advection)

# Initial ocean state from ECCO state estimate
set!(ocean.model, T=ECCOMetadata(:temperature; dates=first(dates)),
                  S=ECCOMetadata(:salinity; dates=first(dates)))

# Atmospheric model
radiation  = Radiation(arch)

# Adding a tidal potential
tidal_potential = FieldTimeSeries("tidal_potential.jld2", "Φ"; architecture=GPU(), backend=InMemory(41))
tidal_potential = FieldTimeSeries("tidal_potential.jld2", "Φ"; architecture=GPU(), backend=InMemory(41), boundary_conditions=FieldBoundaryConditions(tidal_potential.grid, (Center, Center, Nothing)))

atmosphere = JRA55PrescribedAtmosphere(arch; tidal_potential, backend=JRA55NetCDFBackend(41))

# Coupled model and simulation
solver_stop_criteria = FixedIterations(10) # note: more iterations = more accurate
atmosphere_ocean_flux_formulation = SimilarityTheoryFluxes(; solver_stop_criteria)
interfaces = ComponentInterfaces(atmosphere, ocean; radiation, atmosphere_ocean_flux_formulation)
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation, interfaces)
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


checkpointer = Checkpointer(ocean.model,
                            prefix = prefix,
                            schedule = TimeInterval(30days),
                            overwrite_existing = true)

# Run the simulation
run!(simulation)

