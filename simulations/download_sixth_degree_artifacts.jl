#=
Download cached 1/6° bathymetry and ECCO2 initial-condition artifacts
from the GB25Artifacts release on GitHub.

These files are the inputs for `ocean_climate_model_init`:

  - bathymetry_sixth_degree.jld2          ≈ 21 MB
  - ecco2_initial_conditions_sixth_degree.jld2  ≈ 811 MB

Both files are downloaded into the project root (next to Project.toml)
and are skipped if already present.

Run:
    julia --project=. simulations/download_sixth_degree_artifacts.jl
=#

using Downloads

const RELEASE_URL = "https://github.com/glwagner/GB25Artifacts/releases/download/v1.0"

const ARTIFACTS = (
    "bathymetry_sixth_degree.jld2",
    "ecco2_initial_conditions_sixth_degree.jld2",
)

function download_artifact_if_missing(name::AbstractString, dest_dir::AbstractString)
    dest = joinpath(dest_dir, name)
    if isfile(dest)
        @info "$name already present at $dest — skipping"
        return dest
    end
    url = RELEASE_URL * "/" * name
    @info "Downloading $name from $url"
    Downloads.download(url, dest)
    @info "Downloaded $name → $dest"
    return dest
end

function main()
    project_root = abspath(joinpath(@__DIR__, ".."))
    for name in ARTIFACTS
        download_artifact_if_missing(name, project_root)
    end
    return nothing
end

main()
