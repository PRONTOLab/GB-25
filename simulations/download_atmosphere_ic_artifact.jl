#=
Download the atmosphere initial-condition artifact from Dropbox.

  atmosphere_coarsened_1536x768x64.jld2  ≈ 5.4 GB

This is a 1536×768×64 coarsened checkpoint (2:1 aspect preserving
zonal resolution) from a 1/8° (2880×1280×64)
mixed-phase NE-1M microphysics spinup (ExplicitTimeStepping, Δt=1 s,
30 sim minutes from a 1° dry 14-day spinup). Contains prognostic fields
(ρ, ρu, ρv, ρw, ρθ, ρqᵛ) plus microphysical diagnostics (cloud liquid,
cloud ice, vapor).

The 768×768×64 size is chosen so that on-device interpolation to the
target grid lands on clean integer multiples of the per-GPU shard size.

The file is written to `simulations/initial_conditions/` next to this
script and is skipped if already present.

Run:
    julia --project -O0 simulations/download_atmosphere_ic_artifact.jl
=#

using Downloads

const ARTIFACT_NAME = "atmosphere_coarsened_1536x768x64.jld2"
const ARTIFACT_URL  = "https://www.dropbox.com/scl/fi/xxydkjyquq6ryxhohfsc9/atmosphere_coarsened_1536x768x64.jld2?rlkey=5dd443o583kunnwsl6er75yb5&dl=1"

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
    download_artifact_if_missing(ARTIFACT_NAME, ARTIFACT_URL, dest_dir)
    return nothing
end

main()
