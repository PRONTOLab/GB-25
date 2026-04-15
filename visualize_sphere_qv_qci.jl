# Two-panel sphere viz: surface qv + mid-level qci.
# Run: ITER=9000 julia --project=. visualize_sphere_qv_qci.jl

using JLD2
using Oceananigans
using CairoMakie

iter_num = parse(Int, get(ENV, "ITER", "9000"))
ic_dir = joinpath(@__DIR__, "simulations", "initial_conditions")
path = joinpath(ic_dir, "twentyfourth_continued_iter$(iter_num)_assembled.jld2")
isfile(path) || error("Assembled file not found: $path")

Nλ_global = 8640
Nφ_global = 3840
Nz = 64
FT = Float32
k_mid = 15  # ~7 km altitude (Δz=469 m)

@info "Loading slices…" path k_mid
ρ_sfc, ρqv_sfc, ρ_mid, ρqci_mid = JLD2.jldopen(path, "r") do f
    (f["ρ"][:, :, 1], f["ρqᵛ"][:, :, 1],
     f["ρ"][:, :, k_mid + 1], f["micro_ρqᶜⁱ"][:, :, k_mid + 1])
end

safe_sfc  = max.(ρ_sfc,  FT(0.001))
safe_mid  = max.(ρ_mid,  FT(0.001))
qv_sfc    = ρqv_sfc  ./ safe_sfc .* FT(1000)   # g/kg
qci_mid2D = ρqci_mid ./ safe_mid .* FT(1000)   # g/kg

grid = LatitudeLongitudeGrid(CPU();
    size = (Nλ_global, Nφ_global, Nz),
    halo = (4, 4, 4),
    latitude = (-80, 80),
    longitude = (0, 360),
    z = (0, 30e3))

qv_field  = CenterField(grid)
qci_field = CenterField(grid)
Oceananigans.interior(qv_field)[:, :, 1]  .= FT.(qv_sfc)
Oceananigans.interior(qci_field)[:, :, 1] .= FT.(qci_mid2D)

@info "Rendering 2-panel sphere (tight layout)…"

k_render = 1
elev = 0.4
azim = 1.2

# Smaller figure + smaller colgap/protrusions pulls spheres closer
fig = Figure(size = (1100, 620))

ax1 = Axis3(fig[1, 1]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax1); hidespines!(ax1)
plt1 = surface!(ax1, view(qv_field, :, :, k_render);
                colormap = :tempo, colorrange = (0, 50))
Colorbar(fig[2, 1], plt1; vertical = false, width = Relative(0.7),
         label = "qv surface [g/kg]")

ax2 = Axis3(fig[1, 2]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax2); hidespines!(ax2)
plt2 = surface!(ax2, view(qci_field, :, :, k_render);
                colormap = Reverse(:dense), colorrange = (0, 10))
Colorbar(fig[2, 2], plt2; vertical = false, width = Relative(0.7),
         label = "qci at k=$k_mid (~7 km) [g/kg]")

sim_time_h = iter_num * 0.8 / 3600
Label(fig[0, :], "1/24° continuation — iter $iter_num, t=$(round(sim_time_h; digits=2))h sim", fontsize = 18)

# Negative column gap + tight rows pulls panels together
colgap!(fig.layout, -200)
rowgap!(fig.layout, 1, -40)
rowsize!(fig.layout, 1, Relative(0.85))

outpath = joinpath(@__DIR__, "sphere_qv_qci_iter$(iter_num).png")
save(outpath, fig; px_per_unit = 3)  # px_per_unit=3 for faster render vs 4
@info "Saved" outpath
