#=
CPU smoke test for `ocean_climate_model_init`.

Builds the coupled model on a single CPU process at a tiny resolution
using the cached 1/6° artifacts. Reports T, S extrema after `interpolate!`.

Run:
    julia --project=. simulations/test_ocean_climate_model_cpu.jl
=#

using Dates
@info "[$(now())] starting CPU smoke test"

using GordonBell25
using Oceananigans
using Oceananigans.Architectures: CPU
using Printf

@info "[$(now())] building model on CPU at resolution=4, Nz=10..."
model = GordonBell25.ocean_climate_model_init(CPU();
                                              resolution = 4,
                                              Nz         = 10,
                                              Δt         = 60.0)
@info "[$(now())] model built"

ocean_model = model.ocean.model
T = ocean_model.tracers.T
S = ocean_model.tracers.S

Tmin, Tmax = extrema(interior(T))
Smin, Smax = extrema(interior(S))

@info @sprintf("T range: [%.2f, %.2f]  (expect roughly -2..32 °C)", Tmin, Tmax)
@info @sprintf("S range: [%.2f, %.2f]  (expect roughly 0..40 g/kg)", Smin, Smax)

bh = ocean_model.grid.immersed_boundary.bottom_height
bhmin, bhmax = extrema(interior(bh))
@info @sprintf("bottom_height range: [%.2f, %.2f] m", bhmin, bhmax)

@info "[$(now())] grid size: $(size(ocean_model.grid))"
@info "[$(now())] CPU smoke test PASSED"
