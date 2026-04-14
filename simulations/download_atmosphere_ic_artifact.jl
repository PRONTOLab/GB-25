#=
Download the atmosphere initial-condition artifacts from Dropbox.

  atmosphere_coarsened_1536x768x64.jld2   ≈ 5.4 GB
  atmosphere_no_microphysics_1deg_14day.jld2  (1° dry spinup, 360×160×64)

The first is a 1536×768×64 coarsened checkpoint (2:1 aspect preserving
zonal resolution) from a 1/8° (2880×1280×64) mixed-phase NE-1M
microphysics spinup (ExplicitTimeStepping, Δt=1 s, 30 sim minutes from a
1° dry 14-day spinup). Contains prognostic fields (ρ, ρu, ρv, ρw, ρθ, ρqᵛ)
plus microphysical diagnostics (cloud liquid, cloud ice, vapor).

The second is the 1° 14-day dry spinup (no microphysics) at 360×160×64,
used as the initial condition for 1°→fine upsampling tests on both the
vanilla CUDA reproducer (simulations/atmosphere_eighth_from_1deg_cuda.jl)
and the Reactant/sharded path (sharding/sharded_atmosphere_simulation_run.jl).

Files are written to `simulations/initial_conditions/` next to this
script and skipped if already present.

Run:
    julia --project -O0 simulations/download_atmosphere_ic_artifact.jl
    # or to fetch a single file:
    julia --project -O0 simulations/download_atmosphere_ic_artifact.jl \
        atmosphere_no_microphysics_1deg_14day.jld2
=#

using Downloads

const ARTIFACTS = [
    # (name = "atmosphere_coarsened_1536x768x64.jld2",
    #  url  = "https://www.dropbox.com/scl/fi/xxydkjyquq6ryxhohfsc9/atmosphere_coarsened_1536x768x64.jld2?rlkey=5dd443o583kunnwsl6er75yb5&dl=1"),
    # (name = "atmosphere_no_microphysics_1deg_14day.jld2",
    #  url  = "https://www.dropbox.com/scl/fi/0w9hvr8dol7ferfrrn7mj/atmosphere_no_microphysics_1deg_14day.jld2?rlkey=htnm5b8wy89jrt0eu67cbilcc&dl=1"),
    (name = "checkpoint_step_008193.jld2",
     url  = "https://www.dropbox.com/scl/fi/opxpmprauvceg1gr6ew84/checkpoint_step_008193.jld2?rlkey=ahabdl1h74b4vs0m1n400jxcc&st=v7r7wxcg&dl=0"),
]

function download_artifact_if_missing(name::AbstractString, url::AbstractString, dest_dir::AbstractString)
    mkpath(dest_dir)
    dest = joinpath(dest_dir, name)
    if isfile(dest)
        @info "$name already present at $dest — skipping"
        return dest
    end
    @info "Downloading $name from $url"
    Downloads.download(url, dest)
    @info "Downloaded $name → $dest ($(filesize(dest)) bytes)"
    return dest
end

function main()
    dest_dir = joinpath(@__DIR__, "initial_conditions")
    requested = isempty(ARGS) ? [a.name for a in ARTIFACTS] : ARGS
    for a in ARTIFACTS
        a.name in requested || continue
        download_artifact_if_missing(a.name, a.url, dest_dir)
    end
    return nothing
end

main()
