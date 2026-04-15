# Visualize the latest 1/24° continued checkpoint on the sphere.
# Loads from the assembled JLD2 (fast: no per-rank glue needed).
#
# Run: julia --project=. visualize_sphere_24th_continued.jl

using JLD2
using Oceananigans
using CairoMakie

# ── Config ────────────────────────────────────────────────────────────
iter_num = parse(Int, get(ENV, "ITER", "9000"))
ic_dir = joinpath(@__DIR__, "simulations", "initial_conditions")
path = joinpath(ic_dir, "twentyfourth_continued_iter$(iter_num)_assembled.jld2")
isfile(path) || error("Assembled file not found: $path")

Nλ_global = 8640
Nφ_global = 3840
Nz = 64
FT = Float32

# ── Load fields (surface only needed) ─────────────────────────────────
# Reading [:, :, 1] reduces memory to a single k-slice per field.
@info "Loading surface slices from assembled file…" path
ρ, ρθ, ρqv, ρu = JLD2.jldopen(path, "r") do f
    (f["ρ"][:, :, 1], f["ρθ"][:, :, 1], f["ρqᵛ"][:, :, 1], f["ρu"][:, :, 1])
end
@info "Loaded" size_ρ=size(ρ) size_ρθ=size(ρθ) size_ρqv=size(ρqv) size_ρu=size(ρu)

# ── Derive fields ─────────────────────────────────────────────────────
safe_ρ = max.(ρ, FT(0.001))
θ    = ρθ ./ safe_ρ
qv   = ρqv ./ safe_ρ .* FT(1000)  # g/kg
u    = ρu ./ safe_ρ
wspd = abs.(u)  # approximate wind speed from zonal component (ρv assembled is Nφ+1, skip)

# ── Build CPU grid + fields (single-layer) ────────────────────────────
grid = LatitudeLongitudeGrid(CPU();
    size = (Nλ_global, Nφ_global, Nz),
    halo = (4, 4, 4),
    latitude = (-80, 80),
    longitude = (0, 360),
    z = (0, 30e3))

θ_field    = CenterField(grid)
qv_field   = CenterField(grid)
wspd_field = CenterField(grid)

@info "Filling field interior (surface slice only)…"
# Broadcast the 2D slice into all z-levels of each field; we only view k=1 during render.
Oceananigans.interior(θ_field)[:, :, 1]    .= FT.(θ)
Oceananigans.interior(qv_field)[:, :, 1]   .= FT.(qv)
Oceananigans.interior(wspd_field)[:, :, 1] .= FT.(wspd)

# ── Render ────────────────────────────────────────────────────────────
@info "Rendering 3-panel sphere…"

k_sfc = 1
elev = 0.4
azim = 1.2

fig = Figure(size = (1800, 700))

ax1 = Axis3(fig[1, 1]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax1); hidespines!(ax1)
plt1 = surface!(ax1, view(θ_field, :, :, k_sfc); colormap = Reverse(:RdYlBu), colorrange = (250, 320))
Colorbar(fig[2, 1], plt1; vertical = false, width = Relative(0.6), label = "θ [K]")

ax2 = Axis3(fig[1, 2]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax2); hidespines!(ax2)
plt2 = surface!(ax2, view(qv_field, :, :, k_sfc); colormap = :tempo, colorrange = (0, 34))
Colorbar(fig[2, 2], plt2; vertical = false, width = Relative(0.6), label = "qv [g/kg]")

ax3 = Axis3(fig[1, 3]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax3); hidespines!(ax3)
plt3 = surface!(ax3, view(wspd_field, :, :, k_sfc); colormap = :solar, colorrange = (0, 30))
Colorbar(fig[2, 3], plt3; vertical = false, width = Relative(0.6), label = "Wind speed [m/s]")

sim_time_h = iter_num * 0.8 / 3600
Label(fig[0, :], "1/24° continuation — iter $iter_num, t=$(round(sim_time_h; digits=2))h sim — surface fields on sphere", fontsize = 20)

colgap!(fig.layout, -100)
rowgap!(fig.layout, 1, -50)
rowsize!(fig.layout, 1, Relative(0.8))

outpath = joinpath(@__DIR__, "sphere_twentyfourth_continued_iter$(iter_num).png")
save(outpath, fig; px_per_unit = 4)
@info "Saved" outpath
