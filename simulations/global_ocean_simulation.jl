using ClimaOcean
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using ClimaOcean.ECCO
using ClimaOcean.JRA55
using Printf
using Reactant
Reactant.allowscalar(true)

# Runs one time step of a coupled atmosphere/Ocean global simulation using Reactant.

# the following time steps need to be decreased as the resolution is refined
arch = Oceananigans.Architectures.ReactantState()
Δt₁ = 30seconds # small initial spinup time step
Δt₂ = 10minutes # larger time step for after initial spinup
target_dir = "." # where to save output

#####
##### Grid
#####

resolution = 2 # in degrees, set eventually to 1//4
Nx = convert(Int, 360 / resolution)
Ny = convert(Int, 180 / resolution)
Nz = 20 # number of vertical levels, set eventually to 60

z = exponential_z_faces(; Nz, depth=6200)
zᴺ = z[end-1]
z = MutableVerticalDiscretization(z)
grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z)
bottom_height = regrid_bathymetry(grid; minimum_depth=15, major_basins=1)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

#####
##### Prognostic Ocean model
#####

momentum_advection = WENOVectorInvariant(order=5)
tracer_advection   = WENO(order=5)
free_surface = SplitExplicitFreeSurface(grid; substeps=70)

ocean = ocean_simulation(
    grid; 
    Δt = Δt₁,
    vertical_coordinate = Oceananigans.Models.HydrostaticFreeSurfaceModels.ZStar(),
    tracers = (:T, :S, :e, :C),
    momentum_advection,
    tracer_advection,
    free_surface,
)

dataset = ECCO4Monthly()
T_meta = Metadatum(:temperature; dataset)
S_meta = Metadatum(:salinity; dataset)
initial_C(x, y, z) = z > zᴺ
set!(ocean.model, T=T_meta, S=S_meta, C=initial_C)

@show ocean.model

#####
##### Prescribed Atmosphere
#####

atmosphere_dataset = RepeatYearJRA55()
backend = JRA55NetCDFBackend(40)
atmosphere = JRA55PrescribedAtmosphere(arch; dataset=atmosphere_dataset, backend)
radiation  = Radiation()

#####
##### Coupling
#####
 
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

compiled_time_step! = @compile raise=true sync=true Oceananigans.TimeSteppers.time_step!(coupled_model, Δt₁)
compiled_time_step!(coupled_model, Δt₁)