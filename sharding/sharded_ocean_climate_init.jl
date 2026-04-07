#=
Sharded `Distributed{ReactantState}` smoke test for `ocean_climate_model_init`.

Goal: confirm that the new constructor (which loads bathymetry + T, S from
the cached 1/6° JLD2 artifacts and `interpolate!`s onto the target grid)
builds, compiles `first_time_step!`, and runs it once — both for a single
Reactant device and for the multi-device sharded case.

Resolution defaults to 2° (small enough that compile + first step finishes
quickly); override with the `RESOLUTION` env var if you want to push it.

Run (single device):
    LD_PRELOAD=.../libcrypto.so.3 julia +1.11.9 --project=. \
        sharding/sharded_ocean_climate_init.jl

Run (multi-device, e.g. 8 GPUs via Reactant.Distributed):
    LD_PRELOAD=.../libcrypto.so.3 julia +1.11.9 --project=. \
        sharding/sharded_ocean_climate_init.jl
(Reactant.Distributed.initialize picks up XLA_FLAGS / SLURM env on its own.)
=#

using Dates
@info "[$(now(UTC))] this is when the fun begins"

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: factors, is_distributed_env_present
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant
using Printf

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.preamble()
GordonBell25.initialize(; single_gpu_per_process=false)

# ---- Architecture ----
devarch = Oceananigans.ReactantState()
arch    = devarch

Ndev = if arch isa Oceananigans.ReactantState
    length(Reactant.devices())
else
    comm = MPI.COMM_WORLD
    MPI.Comm_size(comm)
end

@info "[startup] Ndev = $Ndev"

Rx, Ry = factors(Ndev)
if Ndev > 1
    arch = Oceananigans.Distributed(devarch; partition = Partition(Rx, Ry, 1))
    rank = Reactant.Distributed.local_rank()
else
    rank = 0
end

@info "[$rank] Rx,Ry = $Rx,$Ry"

# ---- Resolution sweep / single-resolution run ----
resolution = parse(Float64, get(ENV, "RESOLUTION", "2"))
Nz         = parse(Int,     get(ENV, "NZ",         "20"))
Δt         = parse(Float64, get(ENV, "DT_SECONDS", "30"))

@info "[$rank] Building model: resolution=$resolution, Nz=$Nz, Δt=$Δt"
model = GordonBell25.ocean_climate_model_init(arch;
                                              resolution = resolution,
                                              Nz         = Nz,
                                              Δt         = Δt)
@info "[$rank] model built"
@info "[$rank] grid size: $(size(model.ocean.model.grid))"

# Quick sanity: report (T, S, bottom_height) extrema before compile.
# These calls go through Reactant for sharded fields; if interpolate! into
# a sharded target broke, this is where we'd see it.
T = model.ocean.model.tracers.T
S = model.ocean.model.tracers.S
bh = model.ocean.model.grid.immersed_boundary.bottom_height

try
    Tmin, Tmax = extrema(Array(interior(T)))
    Smin, Smax = extrema(Array(interior(S)))
    bhmin, bhmax = extrema(Array(interior(bh)))
    @info @sprintf("[%d] T:[%.2f,%.2f]  S:[%.2f,%.2f]  bh:[%.1f,%.1f]",
                   rank, Tmin, Tmax, Smin, Smax, bhmin, bhmax)
catch err
    @warn "[$rank] could not extract extrema (sharded field?): $err"
end

# ---- Compile first_time_step! under Reactant ----
@info "[$rank] @compile first_time_step!..."
compile_options = CompileOptions(;
    sync = true, raise = true,
    strip_llvm_debuginfo = true,
)

rfirst! = if devarch isa Oceananigans.ReactantState
    @compile compile_options=compile_options first_time_step!(model)
else
    GordonBell25.first_time_step!
end
@info "[$rank] compiled first_time_step!"

# ---- Run it once ----
@info "[$rank] running first_time_step!..."
@time "[$rank] first time step" rfirst!(model)
@info "[$rank] first_time_step! returned"

@info "[$rank] DONE — sharded init test passed at resolution=$resolution"
