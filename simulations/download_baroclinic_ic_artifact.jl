#=
Download the cached 1/4° baroclinic instability initial-condition artifact
from Dropbox.

  baroclinic_100day_768x768x64.jld2  ≈ 600 MB

The file is the input for the `initial_conditions_path` keyword of
`GordonBell25.baroclinic_instability_model`. It is the result of a
~100-day spin-up at 1/4° resolution (90 days at Nz=10 followed by 10
days at Nz=64, both saved as T, S on the model grid).

The file is written to `simulations/initial_conditions/` next to this
script and is skipped if already present.

Run:
    julia --project -O0 simulations/download_baroclinic_ic_artifact.jl
=#

using Downloads

const ARTIFACT_NAME = "baroclinic_100day_768x768x64.jld2"
const ARTIFACT_URL  = "https://www.dropbox.com/scl/fi/5wuj85ted46vjx11upsrm/baroclinic_100day_768x768x64.jld2?rlkey=1a06b8638ju4tpzvjj8uy7bf2&dl=1"

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
