# Equality check for the sharded upscaling path (no data, no MPI).
#
# Upscales one analytic coarse initial condition onto TWO identically-shaped fine targets —
# a sharded Reactant model and a single-device CPU model — and asserts they agree field by
# field. This is the core invariant of `upscale_prognostic_fields!`: sharding the assembly
# must not change the result. Run under emulated devices (single process):
#
#     XLA_FLAGS=--xla_force_host_platform_device_count=4 julia --project \
#         correctness/correctness_sharded_regrid_simulation_run.jl
#
# which is exactly how the `sharded` CI job launches it (see CompileOrRun.yml).

using GordonBell25
using Oceananigans

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 16,
)

default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type
using Reactant

if !GordonBell25.is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.initialize(; single_gpu_per_process = false)
@show Ndev = length(Reactant.devices())
Rx, Ry = GordonBell25.factors(Ndev)

rarch = Oceananigans.ReactantState()
if Ndev != 1
    rarch = Oceananigans.Distributed(rarch; partition = Partition(Rx, Ry, 1))
end

H = 8
Tx = parsed_args["grid-x"] * Rx
Ty = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]
Nx = Tx - 2H
Ny = Ty - 2H

model_kw = (halo = (H, H, H), Δt = 1e-9)

# Coarse source with an analytic initial condition (no external data): baroclinic T, S plus
# nonzero staggered velocities so the Face-located paths (u, v) are exercised non-trivially.
Sx, Sy = cld(Nx, 4), cld(Ny, 4)
source = GordonBell25.baroclinic_instability_model(CPU(), Sx, Sy, Nz; model_kw...)
GordonBell25.set_baroclinic_instability!(source)
set!(source, u = (λ, φ, z) -> 1e-2 * cosd(φ),
             v = (λ, φ, z) -> 1e-2 * sind(2λ) * cosd(φ))

# Two fine targets of identical shape: sharded (Reactant) and single-device (CPU) reference.
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(CPU(), Nx, Ny, Nz; model_kw...)

@info "Upscaling analytic source onto both targets" Ndev tiling = (Rx, Ry) source = (Sx, Sy, Nz) target = (Nx, Ny, Nz)
r_upscaled = GordonBell25.upscale_prognostic_fields!(rmodel, source)
v_upscaled = GordonBell25.upscale_prognostic_fields!(vmodel, source)
@info "Upscaled fields" sharded = r_upscaled single_device = v_upscaled

# Preservation of equality: the sharded assembly must reproduce the single-device result
# for every field (Center tracers, Face velocities, 2D free-surface). `compare_states`
# throws on the first mismatch when `throw_error=true`, failing the CI job.
rtol = sqrt(eps(default_float_type))
atol = 0
@info "Comparing sharded vs single-device upscaled states (interiors):"
equal = GordonBell25.compare_states(rmodel, vmodel; include_halos = false, throw_error = true, rtol, atol)
equal || error("sharded upscaling does not match single-device upscaling")
@info "PASS: sharded upscaling matches single-device upscaling for all fields."
