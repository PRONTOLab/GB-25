#=
Generate a stand-in initial-condition JLD2 file for the 1/4° moist baroclinic
wave model by applying the analytic IC (`set_moist_baroclinic_wave!`) on a
fresh CPU model and dumping the prognostic state to disk in the same format
the loader expects (`set_moist_baroclinic_wave_from_file!`).

The point is to exercise the `initial_conditions_path` code path before any
real spin-up checkpoint exists. Once the spin-up data is ready, drop the
real file in at the same path and the loader will pick it up unchanged.

Run:
    julia --project -O0 simulations/generate_analytic_atmosphere_ic.jl
=#

using Dates
using JLD2
using Oceananigans
using Oceananigans.Architectures: CPU
using Breeze.AtmosphereModels: dynamics_density
using GordonBell25

Oceananigans.defaults.FloatType = Float32

# 1/4° atmosphere defaults — match the model's grid kwargs.
const Nλ = 1440
const Nφ = 640
const Nz = 64
const H  = 30e3
const Δt = 15.0    # 60·(360/Nλ); the loader doesn't care, this is metadata only.

@info "Building 1/4° moist_baroclinic_wave_model on CPU (microphysics=nothing for IC dump)..." now(UTC)
arch  = CPU()
model = GordonBell25.moist_baroclinic_wave_model(arch;
                                                 Nλ = Nλ, Nφ = Nφ, Nz = Nz, H = H,
                                                 Δt = Δt,
                                                 with_microphysics = false,
                                                 halo = (8, 8, 8))

@info "Snapshotting analytic IC fields"
ρ   = Array(interior(dynamics_density(model.dynamics)))
ρu  = Array(interior(model.momentum.ρu))
ρv  = Array(interior(model.momentum.ρv))
ρw  = Array(interior(model.momentum.ρw))
ρθ  = Array(interior(model.formulation.potential_temperature_density))
ρqv = Array(interior(model.moisture_density))

@info "Field shapes" size(ρ) size(ρu) size(ρv) size(ρw) size(ρθ) size(ρqv)
@info "Field extrema" extrema(ρ) extrema(ρu) extrema(ρθ) extrema(ρqv)

outdir = joinpath(@__DIR__, "initial_conditions")
mkpath(outdir)
outfile = joinpath(outdir, "atmosphere_ic_quarter_degree.jld2")

@info "Writing $outfile"
jldsave(outfile;
    ρ, ρu, ρv, ρw, ρθ, ρqᵛ = ρqv,
    Nλ, Nφ, Nz, H,
    Δt        = Δt,
    time      = 0.0,
    iteration = 0,
)
@info "Wrote $(filesize(outfile)) bytes" outfile
