using Dates
using Printf
using CUDA
using JLD2
using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials

# ---------------------------------------------------------------------------
# Load 1/4° checkpoint
# ---------------------------------------------------------------------------
checkpoint_path = if length(ARGS) >= 1
    ARGS[1]
else
    cpdir = joinpath(@__DIR__, "checkpoints")
    files = filter(f -> startswith(f, "ocean_spinup_quarter_degree_180day_") && endswith(f, ".jld2"),
                   readdir(cpdir))
    isempty(files) && error("No quarter-degree checkpoint found in $cpdir. Run ocean_spinup_vanilla.jl first.")
    joinpath(cpdir, sort(files)[end])
end

@info "Loading 1/4° checkpoint" checkpoint_path
Nx_src, Ny_src, Nz_src, u_data, v_data, T_data, S_data = JLD2.jldopen(checkpoint_path, "r") do f
    (f["Nx"], f["Ny"], f["Nz"], f["u"], f["v"], f["T"], f["S"])
end
@info "Loaded" Nx_src Ny_src Nz_src size(T_data) extrema(T_data) extrema(S_data) extrema(u_data) extrema(v_data)

# ---------------------------------------------------------------------------
# Build 1/10° model
# ---------------------------------------------------------------------------
const Nx = 3840  # 1/10°
const Ny = 1920
const Nz = 64
const Δt         = 30  # seconds; CFL ≈ 0.2 at U_max=12 m/s, Δx_min=1829 m
const total_time = 15days
const Ninner     = 64
const Nouter     = ceil(Int, total_time / (Ninner * Δt))
const Nt         = Ninner * Nouter

@info "Building 1/10° model on CUDA GPU..." now(UTC)

grid = LatitudeLongitudeGrid(GPU();
    size = (Nx, Ny, Nz),
    halo = (8, 8, 8),
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)

model = HydrostaticFreeSurfaceModel(;
    grid,
    free_surface = SplitExplicitFreeSurface(substeps=60),
    buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState(Float64)),
    closure = nothing,
    coriolis = HydrostaticSphericalCoriolis(),
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    tracers = (:T, :S),
)

# ---------------------------------------------------------------------------
# Interpolate 1/4° → 1/10°
# ---------------------------------------------------------------------------
@info "Interpolating 1/4° fields to 1/10° grid..." now(UTC)

source_grid = LatitudeLongitudeGrid(GPU();
    size = (Nx_src, Ny_src, Nz_src),
    halo = (1, 1, 1),
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)

FT = Float64

# Tracers (Center, Center, Center)
T_src = CenterField(source_grid)
S_src = CenterField(source_grid)
set!(T_src, FT.(T_data))
set!(S_src, FT.(S_data))
Oceananigans.BoundaryConditions.fill_halo_regions!(T_src)
Oceananigans.BoundaryConditions.fill_halo_regions!(S_src)

Oceananigans.Fields.interpolate!(model.tracers.T, T_src)
Oceananigans.Fields.interpolate!(model.tracers.S, S_src)
@info "Interpolated T, S" extrema_T=extrema(Array(interior(model.tracers.T))) extrema_S=extrema(Array(interior(model.tracers.S)))

# Velocities
u_src = XFaceField(source_grid)
v_src = YFaceField(source_grid)
set!(u_src, FT.(u_data))
set!(v_src, FT.(v_data))
Oceananigans.BoundaryConditions.fill_halo_regions!(u_src)
Oceananigans.BoundaryConditions.fill_halo_regions!(v_src)

Oceananigans.Fields.interpolate!(model.velocities.u, u_src)
Oceananigans.Fields.interpolate!(model.velocities.v, v_src)
@info "Interpolated u, v" extrema_u=extrema(Array(interior(model.velocities.u))) extrema_v=extrema(Array(interior(model.velocities.v)))

@show model

model.clock.last_Δt = Δt

# ---------------------------------------------------------------------------
# Time-step 15 days
# ---------------------------------------------------------------------------
@info "First time step..." now(UTC)
@time "first time step" Oceananigans.TimeSteppers.first_time_step!(model, Δt)

@info "Stepping $Nt steps at Δt=$(Δt)s ($(Nouter) outer × $(Ninner) inner) → $(round(Nt*Δt/86400, digits=2)) days..." now(UTC)
sim_seconds_per_loop = Ninner * Δt
wall_start = time_ns()

const Δx_min = Oceananigans.Grids.minimum_xspacing(model.grid)
const Δy_min = Oceananigans.Grids.minimum_yspacing(model.grid)
const Δh_min = min(Δx_min, Δy_min)
@info "Grid horizontal spacing" Δx_min Δy_min Δh_min

for k in 1:Nouter
    t0 = time_ns()
    for _ in 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    CUDA.synchronize()
    wall      = (time_ns() - t0) / 1e9
    total_w   = (time_ns() - wall_start) / 1e9
    sim_days  = k * sim_seconds_per_loop / 86400
    sypd_inst = (sim_seconds_per_loop / wall) / 365.25
    sypd_avg  = (k * sim_seconds_per_loop / total_w) / 365.25

    umax = maximum(abs, Array(interior(model.velocities.u)))
    vmax = maximum(abs, Array(interior(model.velocities.v)))
    wmax = maximum(abs, Array(interior(model.velocities.w)))
    Umax = max(umax, vmax)
    cfl  = Umax * Δt / Δh_min

    @info @sprintf("loop %d/%d  wall=%.3fs  sim=%.3f days  SYPD(avg)=%.4f  |u|=%.4g  |v|=%.4g  |w|=%.4g  CFL=%.4f",
                   k, Nouter, wall, sim_days, sypd_avg, umax, vmax, wmax, cfl)

    if !isfinite(umax) || !isfinite(vmax) || !isfinite(wmax)
        error("Velocity went non-finite at loop $k (sim=$(sim_days) days). Aborting.")
    end
    if cfl > 0.7
        error(@sprintf("CFL=%.3f > 0.7 at loop %d (sim=%.3f days). Aborting before blowup.",
                       cfl, k, sim_days))
    end
end

@info "Done stepping" now(UTC) model.clock

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------
checkpoint_dir = joinpath(@__DIR__, "checkpoints")
mkpath(checkpoint_dir)
jobid = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH-MM-SS")
outpath = joinpath(checkpoint_dir, "ocean_tenth_degree_15day_$(jobid).jld2")

@info "Saving u, v, T, S checkpoint" outpath
@time "checkpoint save" begin
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    T = Array(interior(model.tracers.T))
    S = Array(interior(model.tracers.S))
    JLD2.jldsave(outpath;
                 u, v, T, S,
                 Nx, Ny, Nz,
                 Δt = Δt,
                 time = model.clock.time,
                 iteration = model.clock.iteration)
end
@info "Saved" outpath filesize(outpath)

@info "Done!" now(UTC)
