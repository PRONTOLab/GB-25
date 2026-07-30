# Continue the 1/10° ocean spinup at 1/24° across 8 GPUs using vanilla
# Oceananigans.Distributed + MPI (no Reactant; Reactant poisons CUDA.jl kernels —
# see SIMULATION_NOTES.md).
#
# Pipeline position:
#   1. ocean_spinup_vanilla.jl            → 1/4° checkpoint (u, v, T, S)
#   2. ocean_tenth_degree_from_quarter.jl → 1/10° checkpoint (u, v, T, S)
#   3. THIS SCRIPT                        → 1/24° spinup on 8 H100
#
# Launch (after stripping CUDA toolkit path, see SIMULATION_NOTES.md):
#
#   LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
#     mpiexecjl -n 8 julia -O0 --project=. simulations/ocean_24th_degree_8gpu.jl \
#     path/to/ocean_tenth_degree_15day.jld2 \
#     > simulations/logs/run_24th_$(date -u +%Y-%m-%dT%H-%M-%S).log 2>&1
#
# The checkpoint path is the single positional argument. If omitted, the script
# uses the most recent `ocean_tenth_degree_15day_*.jld2` in simulations/checkpoints/.

using MPI
MPI.Init()

using Dates
using Printf
using CUDA
using JLD2
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using SeawaterPolynomials

Oceananigans.defaults.FloatType = Float64
const FT = Float64

# ---------------------------------------------------------------------------
# Architecture: 8 GPUs, 4x2 horizontal partition
# ---------------------------------------------------------------------------
const Nx, Ny, Nz = 9216, 4608, 64   # 1/24°
const Δt         = 12               # seconds; CFL ≈ 0.2 at U_max=12 m/s, Δx_min=753 m
const total_time = 15days
const Ninner     = 64
const Nouter     = ceil(Int, total_time / (Ninner * Δt))
const Nt         = Ninner * Nouter

arch = Distributed(GPU(); partition=Partition(4, 2, 1))
rank = MPI.Comm_rank(arch.communicator)
root = rank == 0

root && @info "Launching 1/24° run" now(UTC) Nx Ny Nz Δt total_time Nouter

# ---------------------------------------------------------------------------
# Load the 1/10° checkpoint path (rank 0 resolves, broadcast to all)
# ---------------------------------------------------------------------------
checkpoint_path = if length(ARGS) >= 1
    ARGS[1]
else
    cpdir = joinpath(@__DIR__, "checkpoints")
    files = filter(f -> startswith(f, "ocean_tenth_degree_15day_") && endswith(f, ".jld2"),
                   isdir(cpdir) ? readdir(cpdir) : String[])
    isempty(files) && error("No tenth-degree checkpoint in $cpdir and no path given. " *
                            "Download from Dropbox (see NEXT_STEPS_8xH100.md) or pass a path.")
    joinpath(cpdir, sort(files)[end])
end
root && @info "Checkpoint" checkpoint_path

# ---------------------------------------------------------------------------
# Every rank loads full checkpoint (5 GB per field × 4 fields = 20 GB host per rank,
# fine — 80 GB H100 has plenty). interpolate! will populate each rank's local
# partition of the 1/24° target from the full 1/10° source on its GPU.
# ---------------------------------------------------------------------------
root && @info "Loading tenth-degree checkpoint..." now(UTC)
Nx_src, Ny_src, Nz_src, u_data, v_data, T_data, S_data = JLD2.jldopen(checkpoint_path, "r") do f
    (f["Nx"], f["Ny"], f["Nz"], f["u"], f["v"], f["T"], f["S"])
end
root && @info "Loaded" Nx_src Ny_src Nz_src extrema(T_data) extrema(u_data)

# ---------------------------------------------------------------------------
# Build target 1/24° model on the distributed architecture
# ---------------------------------------------------------------------------
root && @info "Building 1/24° distributed model..." now(UTC)

grid = LatitudeLongitudeGrid(arch;
    size = (Nx, Ny, Nz),
    halo = (8, 8, 8),
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)

model = HydrostaticFreeSurfaceModel(;
    grid,
    free_surface = SplitExplicitFreeSurface(substeps=60),
    buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState(FT)),
    closure = nothing,
    coriolis = HydrostaticSphericalCoriolis(),
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    tracers = (:T, :S),
)

model.clock.last_Δt = Δt

# ---------------------------------------------------------------------------
# Interpolate 1/10° → 1/24° on GPU.
#
# Source grid MUST live on the same architecture as the target. For a
# distributed target, using a plain-GPU source (not distributed) causes
# interpolate!'s architecture check to pass on each rank individually because
# the target's child_architecture is GPU(). If that assumption ever changes,
# build `source_grid` on `arch` instead and broadcast the source data via
# `set!` the same way.
# ---------------------------------------------------------------------------
root && @info "Interpolating 1/10° fields to 1/24° grid..." now(UTC)

source_grid = LatitudeLongitudeGrid(GPU();
    size = (Nx_src, Ny_src, Nz_src),
    halo = (1, 1, 1),
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)

T_src = CenterField(source_grid); set!(T_src, FT.(T_data))
S_src = CenterField(source_grid); set!(S_src, FT.(S_data))
u_src = XFaceField(source_grid);  set!(u_src, FT.(u_data))
v_src = YFaceField(source_grid);  set!(v_src, FT.(v_data))

for src in (T_src, S_src, u_src, v_src)
    Oceananigans.BoundaryConditions.fill_halo_regions!(src)
end

Oceananigans.Fields.interpolate!(model.tracers.T,    T_src)
Oceananigans.Fields.interpolate!(model.tracers.S,    S_src)
Oceananigans.Fields.interpolate!(model.velocities.u, u_src)
Oceananigans.Fields.interpolate!(model.velocities.v, v_src)

# Free source memory before time-stepping
T_src = S_src = u_src = v_src = nothing
u_data = v_data = T_data = S_data = nothing
GC.gc()
CUDA.reclaim()

root && @info "Interpolation complete" now(UTC)

# ---------------------------------------------------------------------------
# Time step
# ---------------------------------------------------------------------------
root && @info "First time step..." now(UTC)
@time "first time step" Oceananigans.TimeSteppers.first_time_step!(model, Δt)

root && @info "Stepping $Nt steps at Δt=$(Δt)s ($(Nouter) × $(Ninner)) → $(round(Nt*Δt/86400, digits=2)) days..."
sim_seconds_per_loop = Ninner * Δt
wall_start = time_ns()

Δx_min = Oceananigans.Grids.minimum_xspacing(model.grid)
Δy_min = Oceananigans.Grids.minimum_yspacing(model.grid)
Δh_min = min(Δx_min, Δy_min)
root && @info "Grid horizontal spacing" Δx_min Δy_min Δh_min

for k in 1:Nouter
    t0 = time_ns()
    for _ in 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    CUDA.synchronize()

    # Rank-0 diagnostics — each rank only sees its partition's extrema, so we
    # Allreduce to get the global velocity max for CFL.
    local_umax = maximum(abs, Array(interior(model.velocities.u)))
    local_vmax = maximum(abs, Array(interior(model.velocities.v)))
    local_wmax = maximum(abs, Array(interior(model.velocities.w)))
    umax = MPI.Allreduce(local_umax, MPI.MAX, arch.communicator)
    vmax = MPI.Allreduce(local_vmax, MPI.MAX, arch.communicator)
    wmax = MPI.Allreduce(local_wmax, MPI.MAX, arch.communicator)
    Umax = max(umax, vmax)
    cfl  = Umax * Δt / Δh_min

    if root
        wall    = (time_ns() - t0) / 1e9
        total_w = (time_ns() - wall_start) / 1e9
        sim_days  = k * sim_seconds_per_loop / 86400
        sypd_avg  = (k * sim_seconds_per_loop / total_w) / 365.25
        @info @sprintf("loop %d/%d  wall=%.3fs  sim=%.3f days  SYPD(avg)=%.4f  |u|=%.4g  |v|=%.4g  |w|=%.4g  CFL=%.4f",
                       k, Nouter, wall, sim_days, sypd_avg, umax, vmax, wmax, cfl)
    end

    if !isfinite(umax) || !isfinite(vmax) || !isfinite(wmax)
        error("Velocity went non-finite at loop $k. Aborting.")
    end
    if cfl > 0.7
        error(@sprintf("CFL=%.3f > 0.7 at loop %d. Aborting before blowup.", cfl, k))
    end
end

root && @info "Done stepping" now(UTC) model.clock

# ---------------------------------------------------------------------------
# Save per-rank partition. Each rank writes its own local slab. Reassembly is
# a post-processing step (or just use these for the next resolution bump).
# ---------------------------------------------------------------------------
checkpoint_dir = joinpath(@__DIR__, "checkpoints")
root && mkpath(checkpoint_dir)
MPI.Barrier(arch.communicator)

jobid = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH-MM-SS")
outpath = joinpath(checkpoint_dir, "ocean_24th_degree_15day_$(jobid)_rank$(lpad(rank,2,'0')).jld2")

@info "[$rank] Saving local partition" outpath
u_local = Array(interior(model.velocities.u))
v_local = Array(interior(model.velocities.v))
T_local = Array(interior(model.tracers.T))
S_local = Array(interior(model.tracers.S))

JLD2.jldsave(outpath;
             u=u_local, v=v_local, T=T_local, S=S_local,
             Nx, Ny, Nz,
             rank, Rx=4, Ry=2,
             partition_x=arch.partition.x, partition_y=arch.partition.y,
             Δt = Δt,
             time = model.clock.time,
             iteration = model.clock.iteration)

MPI.Barrier(arch.communicator)
root && @info "Done!" now(UTC)
