#=
Generate a stand-in initial-condition JLD2 file for the 1/4° baroclinic
instability model by applying the analytic IC (`set_baroclinic_instability!`)
on a fresh CPU model and dumping T, S to disk in the same format the loader
expects (`set_baroclinic_instability_from_file!`).

The point is to exercise the `initial_conditions_path` code path before any
real spin-up checkpoint exists. Once the spin-up data is ready, drop the
real file in at the same path and the loader will pick it up unchanged.

Run:
    julia --project -O0 simulations/generate_analytic_baroclinic_ic.jl
=#

using Dates
using JLD2
using Oceananigans
using Oceananigans.Units
using GordonBell25

Oceananigans.defaults.FloatType = Float64

const Nx = 1536  # 1/4°
const Ny = 768
const Nz = 64
const Δt = 4minutes

@info "Building 1/4° baroclinic_instability_model on CPU (no closure)..." now(UTC)
arch  = CPU()
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz;
                                                  halo=(8, 8, 8), Δt=Δt)

@info "Applying analytic IC via set_baroclinic_instability!..."
GordonBell25._set_baroclinic_instability!(model)

T = Array(interior(model.tracers.T))
S = Array(interior(model.tracers.S))
@info "T, S populated" extrema_T=extrema(T) extrema_S=extrema(S)

outdir = joinpath(@__DIR__, "initial_conditions")
mkpath(outdir)
outfile = joinpath(outdir, "baroclinic_ic_quarter_degree.jld2")

@info "Writing $outfile"
jldsave(outfile;
    T, S,
    Nx, Ny, Nz,
    Δt = Δt,
    time = 0.0,
    iteration = 0,
)
@info "Wrote $(filesize(outfile)) bytes" outfile
