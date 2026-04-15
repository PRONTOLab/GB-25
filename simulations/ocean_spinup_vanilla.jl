using Dates
using Printf
using CUDA
using JLD2
using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials
const Nx = 1536  # 1/4°
const Ny = 768
const Nz = 64
const Δt         = 2minutes
const total_time = 180days
const Ninner     = 64
const Nouter     = ceil(Int, total_time / (Ninner * Δt))
const Nt         = Ninner * Nouter

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
@info "Building 1/4° model on CUDA GPU..." now(UTC)

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
# Initial conditions (broadband random perturbation, strong gradient)
# ---------------------------------------------------------------------------
Tᵢ(λ, φ, z) = 30 * (1 - tanh((abs(φ) - 35) / 8)) / 2 + rand()
Sᵢ(λ, φ, z) = 28 - 5e-3 * z + rand()
set!(model, T=Tᵢ, S=Sᵢ)
@info "Set initial T, S" extrema_T=extrema(Array(interior(model.tracers.T))) extrema_S=extrema(Array(interior(model.tracers.S)))
@show model

model.clock.last_Δt = Δt

# ---------------------------------------------------------------------------
# Time-step
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
checkpoint_path = joinpath(checkpoint_dir, "ocean_spinup_quarter_degree_180day_$(jobid).jld2")

@info "Saving u, v, T, S checkpoint" checkpoint_path
@time "checkpoint save" begin
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    T = Array(interior(model.tracers.T))
    S = Array(interior(model.tracers.S))
    JLD2.jldsave(checkpoint_path;
                 u, v, T, S,
                 Nx, Ny, Nz,
                 Δt = Δt,
                 time = model.clock.time,
                 iteration = model.clock.iteration)
end
@info "Saved" checkpoint_path filesize(checkpoint_path)

@info "Done!" now(UTC)
