using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using CUDA
using Reactant
using Printf

preamble()

# ============================================================
# Configuration
# ============================================================

resolution = 2
Ninner = ConcreteRNumber(100)

# ============================================================
# Build model with bathymetry + ECCO ICs
# ============================================================

@info "Generating model at $(resolution)° with bathymetry + ECCO ICs..."
model = ocean_climate_model_init(ReactantState(); resolution)

GC.gc(true); GC.gc(false); GC.gc(true)

# ============================================================
# Compile
# ============================================================

@info "Compiling first_time_step!..."
t0 = time_ns()
rfirst! = @compile raise=true sync=true first_time_step!(model)
@info @sprintf("first_time_step! compiled in %.1f seconds", 1e-9 * (time_ns() - t0))

@info "Compiling loop! (Ninner=$(Int(Ninner)))..."
t0 = time_ns()
rloop! = @compile raise=true sync=true loop!(model, Ninner)
@info @sprintf("loop! compiled in %.1f seconds", 1e-9 * (time_ns() - t0))

# ============================================================
# Run
# ============================================================

@info "Running first_time_step!..."
t0 = time_ns()
rfirst!(model)
@info @sprintf("first_time_step! done in %.2f seconds", 1e-9 * (time_ns() - t0))

@info "Running loop! ($(Int(Ninner)) steps)..."
t0 = time_ns()
rloop!(model, Ninner)
elapsed = 1e-9 * (time_ns() - t0)
@info @sprintf("loop! done: %d steps in %.2f seconds (%.1f steps/s)", Int(Ninner), elapsed, Int(Ninner) / elapsed)

# Compute SYPD: each step is Δt=30s, so Ninner steps = Ninner*30 seconds of sim time
Δt_sim = 30  # seconds (from ocean_climate_model_init default)
sim_days = Int(Ninner) * Δt_sim / 86400
sypd = sim_days / elapsed * 86400 / 365.25
@info @sprintf("Performance: %.2f SYPD (Δt=%ds)", sypd, Δt_sim)

@info "Done!"
