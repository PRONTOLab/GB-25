using MPI
MPI.Init()

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations: Distributed, Partition
using Oceananigans.Fields: interpolate!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using CUDA
using NCCL

using NumericalEarth
using NumericalEarth.EarthSystemModels.InterfaceComputations: TenUnrolledIterations, ComponentInterfaces

using Dates
using Downloads
using JLD2
using Printf

# ============================================================
# Artifact download (rank 0 only, then barrier)
# ============================================================

const GB25_ARTIFACTS_URL = "https://github.com/glwagner/GB25Artifacts/releases/download/v1.0"

function download_artifact_if_missing(path)
    if !isfile(path)
        url = GB25_ARTIFACTS_URL * "/" * basename(path)
        @info "Downloading $(basename(path)) from $url..."
        Downloads.download(url, path)
        @info "Downloaded $(basename(path)) → $path"
    end
    return path
end

# ============================================================
# Configuration
# ============================================================

resolution = parse(Float64, get(ENV, "RESOLUTION", "0.25"))
Δt_minutes = parse(Float64, get(ENV, "DT_MINUTES", "15"))
Δt         = Δt_minutes * 60
Nz         = 20
stop_days  = parse(Float64, get(ENV, "STOP_DAYS", "60"))
stop_time  = stop_days * days

bathymetry_file = joinpath(@__DIR__, "..", "bathymetry_sixth_degree.jld2")
ic_file         = joinpath(@__DIR__, "..", "ecco2_initial_conditions_sixth_degree.jld2")

output_prefix = "ocean_$(replace(string(resolution), '/' => '-'))deg_ecco"

# ============================================================
# MPI + NCCL
# ============================================================

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

Rx, Ry = 4, 2
@assert Rx * Ry == Nranks

using Oceananigans.DistributedComputations
NCCLDistributed = Base.get_extension(Oceananigans, :OceananigansNCCLExt).NCCLDistributed
arch = NCCLDistributed(GPU(CUDA.CUDABackend()); partition = Partition(Rx, Ry, 1))

rank == 0 && @info "Config: $(resolution)°, Δt=$(Δt_minutes)min, $Nranks GPUs"

# ============================================================
# Grid
# ============================================================

Nx = convert(Int, 384 / resolution)
Ny = convert(Int, 192 / resolution)
z = ExponentialDiscretization(Nz, -4000, 0; scale=1000)

rank == 0 && @info "Grid: $Nx × $Ny × $Nz"
grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(8, 8, 8), z,
                             latitude=(-80, 80), longitude=(0, 360))

# ============================================================
# Bathymetry — interpolate from cached 1/6° file
# ============================================================

# Rank 0 fetches both artifacts (if missing); other ranks wait at the barrier.
if rank == 0
    download_artifact_if_missing(bathymetry_file)
    download_artifact_if_missing(ic_file)
end
MPI.Barrier(comm)

rank == 0 && @info "Loading bathymetry from $bathymetry_file..."
bathy_data = jldopen(bathymetry_file)
h_cached = bathy_data["bottom_height"]
Nx_bathy = size(h_cached, 1)
Ny_bathy = size(h_cached, 2)
close(bathy_data)

topo_2d = (Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded, Oceananigans.Grids.Flat)
cpu_bathy_grid = LatitudeLongitudeGrid(CPU(); topology=topo_2d,
                                       size = (Nx_bathy, Ny_bathy),
                                       halo = (8, 8),
                                       longitude = (0, 360),
                                       latitude = (-80, 80))

source_bh = Field{Center, Center, Nothing}(cpu_bathy_grid)
set!(source_bh, h_cached)
fill_halo_regions!(source_bh)

# Interpolate onto CPU target grid, then set on distributed grid
cpu_bh_grid = LatitudeLongitudeGrid(CPU(); topology=topo_2d,
                                    size = (Nx, Ny),
                                    halo = (8, 8),
                                    longitude = (0, 360),
                                    latitude = (-80, 80))

cpu_bottom_height = Field{Center, Center, Nothing}(cpu_bh_grid)
interpolate!(cpu_bottom_height, source_bh)

bottom_height = Field{Center, Center, Nothing}(grid)
set!(bottom_height, Array(interior(cpu_bottom_height)))

rank == 0 && @info "Creating ImmersedBoundaryGrid..."
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

# ============================================================
# Ocean simulation
# ============================================================

# Compute substeps on CPU grid (avoids MPI issues in free surface constructor)
cpu_grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz), halo=(8,8,8), z,
                                 latitude=(-80,80), longitude=(0,360))
cpu_fs = SplitExplicitFreeSurface(cpu_grid; cfl=0.7, fixed_Δt=Δt)
substeps = length(cpu_fs.substepping.averaging_weights)
free_surface = SplitExplicitFreeSurface(substeps=substeps)

rank == 0 && @info "Ocean simulation: substeps=$substeps"
ocean = ocean_simulation(grid; free_surface, Δt)

# ============================================================
# Initial conditions — interpolate from cached ECCO2
# ============================================================

rank == 0 && @info "Loading ECCO2 initial conditions..."
ic = jldopen(ic_file)
T_cached = ic["T"]
S_cached = ic["S"]
Nx_ic    = ic["Nx"]
Ny_ic    = ic["Ny"]
Nz_ic    = ic["Nz"]
close(ic)

# Interpolate on CPU, then set onto distributed GPU fields
ic_z = ExponentialDiscretization(Nz_ic, -4000, 0; scale=1000)
cpu_ic_src = LatitudeLongitudeGrid(CPU(); size=(Nx_ic, Ny_ic, Nz_ic), halo=(8,8,8),
                                   z=ic_z, latitude=(-80,80), longitude=(0,360))
cpu_ic_dst = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz), halo=(8,8,8),
                                   z=z, latitude=(-80,80), longitude=(0,360))

src_T = Field{Center, Center, Center}(cpu_ic_src); set!(src_T, T_cached); fill_halo_regions!(src_T)
src_S = Field{Center, Center, Center}(cpu_ic_src); set!(src_S, S_cached); fill_halo_regions!(src_S)

dst_T = Field{Center, Center, Center}(cpu_ic_dst); interpolate!(dst_T, src_T)
dst_S = Field{Center, Center, Center}(cpu_ic_dst); interpolate!(dst_S, src_S)

set!(ocean.model, T=Array(interior(dst_T)), S=Array(interior(dst_S)))

rank == 0 && @info "IC set: T=($(minimum(interior(ocean.model.tracers.T))), $(maximum(interior(ocean.model.tracers.T))))"

# ============================================================
# Atmosphere
# ============================================================

zonal_wind(λ, φ) = 4 * sind(2φ)^2 - 2 * exp(-(abs(φ) - 12)^2 / 72)
sunlight(λ, φ) = -200 - 600 * cosd(φ)^2
Tatm(λ, φ, z=0) = 30 * cosd(φ)
qatm(λ, φ) = 0.003 + 0.018 * cosd(φ)  # ~21 g/kg at equator, ~3 g/kg at poles (78% RH)

topo = (Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded, Oceananigans.Grids.Flat)
ag = LatitudeLongitudeGrid(arch; topology=topo, size=(360,180),
                           longitude=(0,360), latitude=(-90,90), z=nothing)
atm = PrescribedAtmosphere(ag, range(0, 1days, length=24))

Ta = Field{Center, Center, Nothing}(ag); set!(Ta, Tatm)
ua = Field{Center, Center, Nothing}(ag); set!(ua, zonal_wind)
Qs = Field{Center, Center, Nothing}(ag); set!(Qs, sunlight)

parent(atm.tracers.T) .= parent(Ta) .+ 273.15
parent(atm.velocities.u) .= parent(ua)
parent(atm.downwelling_radiation.shortwave) .= parent(Qs)
qa = Field{Center, Center, Nothing}(ag); set!(qa, qatm)
parent(atm.tracers.q) .= parent(qa)

radiation = Radiation(arch)
ifc = ComponentInterfaces(atm, ocean; radiation,
    atmosphere_ocean_fluxes=SimilarityTheoryFluxes(; solver_stop_criteria=TenUnrolledIterations()))
coupled_model = OceanOnlyModel(ocean; atmosphere=atm, radiation, interfaces=ifc)

# ============================================================
# Simulation
# ============================================================

simulation = Simulation(coupled_model; Δt, stop_time)

wall_time = Ref(time_ns())
time_start = Ref(0.0)

function progress(sim)
    m = sim.model.ocean.model
    u, v, w = m.velocities
    T = m.tracers.T
    Tmax = maximum(interior(T)); Tmin = minimum(interior(T))
    umax = (maximum(abs, interior(u)), maximum(abs, interior(v)), maximum(abs, interior(w)))
    elapsed = 1e-9 * (time_ns() - wall_time[])
    sim_days = (time(sim) - time_start[]) / 86400
    sypd = (elapsed > 0 && sim_days > 0) ? sim_days / elapsed * 86400 / 365.25 : NaN

    Oceananigans.DistributedComputations.@root @info @sprintf(
        "Time: %s, n: %d, |u|: (%.2e, %.2e, %.2e), T: (%.2f, %.2f), wall: %s, SYPD: %.2f",
        prettytime(sim), iteration(sim), umax..., Tmin, Tmax, prettytime(elapsed), sypd)

    wall_time[] = time_ns()
    time_start[] = time(sim)
end

add_callback!(simulation, progress, IterationInterval(100))

# ============================================================
# Surface output writer
# ============================================================

ocean_model = ocean.model
Nz_model = size(ocean_model.grid, 3)

surface_outputs = (; T = ocean_model.tracers.T,
                     S = ocean_model.tracers.S,
                     u = ocean_model.velocities.u,
                     v = ocean_model.velocities.v)

output_interval = parse(Float64, get(ENV, "OUTPUT_HOURS", "12"))

simulation.output_writers[:surface] = JLD2Writer(ocean_model, surface_outputs;
    filename = output_prefix * "_surface",
    indices = (:, :, Nz_model),
    schedule = TimeInterval(output_interval * hours),
    overwrite_existing = true)

rank == 0 && @info "Surface output every $(output_interval)h → $(output_prefix)_surface"

# ============================================================
# Run
# ============================================================

rank == 0 && @info "Running for $stop_time..."
wall_start = time_ns()
run!(simulation)
wall_total = 1e-9 * (time_ns() - wall_start)
sypd_total = (stop_time / 86400) / wall_total * 86400 / 365.25

rank == 0 && @info @sprintf("Complete! Wall: %s, SYPD: %.2f", prettytime(wall_total), sypd_total)

# ============================================================
# Dump final 3D T, S fields (same format as ECCO2 IC file)
# ============================================================

rank == 0 && @info "Saving final 3D fields..."

# Gather T and S to CPU arrays on each rank, then collect on rank 0
T_local = Array(interior(ocean_model.tracers.T))
S_local = Array(interior(ocean_model.tracers.S))

# Each rank has a local chunk — gather to rank 0
T_gathered = MPI.Gather(T_local, comm; root=0)
S_gathered = MPI.Gather(S_local, comm; root=0)

if rank == 0
    # Reconstruct global arrays from gathered chunks
    # Partition is Rx × Ry; each local chunk is (nx, ny, Nz)
    nx_local = Nx ÷ Rx
    ny_local = Ny ÷ Ry

    T_global = zeros(Float64, Nx, Ny, Nz)
    S_global = zeros(Float64, Nx, Ny, Nz)

    for r in 0:(Nranks-1)
        rx = r % Rx
        ry = r ÷ Rx
        i1 = rx * nx_local + 1
        j1 = ry * ny_local + 1
        offset = r * nx_local * ny_local * Nz
        chunk_T = reshape(T_gathered[offset+1 : offset + nx_local*ny_local*Nz], nx_local, ny_local, Nz)
        chunk_S = reshape(S_gathered[offset+1 : offset + nx_local*ny_local*Nz], nx_local, ny_local, Nz)
        T_global[i1:i1+nx_local-1, j1:j1+ny_local-1, :] .= chunk_T
        S_global[i1:i1+nx_local-1, j1:j1+ny_local-1, :] .= chunk_S
    end

    ic_filename = "$(output_prefix)_final_conditions.jld2"
    jldsave(ic_filename;
            T = Float32.(T_global),
            S = Float32.(S_global),
            Nx, Ny, Nz,
            latitude = (-80, 80),
            longitude = (0, 360),
            date = "after $(Int(stop_days)) days from ECCO2 2023-03-06")

    @info "Saved final conditions to $ic_filename"
end
