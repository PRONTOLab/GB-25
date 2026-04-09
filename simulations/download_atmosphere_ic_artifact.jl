#=
Download the cached 1° atmosphere initial-condition artifact from Dropbox.

  atmosphere_no_microphysics_1deg_14day.jld2  ≈ 85 MB

This is the 14-day spinup of the 1° (Nλ=360, Nφ=160, Nz=64)
moist-baroclinic-wave model with `microphysics=nothing` and Δt=60 s. It is
the input for the `initial_conditions_path` keyword of
`GordonBell25.moist_baroclinic_wave_model`.

The file is written to `simulations/initial_conditions/` next to this
script and is skipped if already present.

Run:
    julia --project -O0 simulations/download_atmosphere_ic_artifact.jl
=#

using Downloads

const ARTIFACT_NAME = "atmosphere_no_microphysics_1deg_14day.jld2"
const ARTIFACT_URL  = "https://www.dropbox.com/scl/fi/0w9hvr8dol7ferfrrn7mj/atmosphere_no_microphysics_1deg_14day.jld2?rlkey=htnm5b8wy89jrt0eu67cbilcc&dl=1"

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
