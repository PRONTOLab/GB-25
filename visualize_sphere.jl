# Visualize 1/8° checkpoint on sphere: surface θ, qv, wind speed.

using JLD2
using Oceananigans
using CairoMakie

ic_path = joinpath(@__DIR__, "simulations", "initial_conditions", "checkpoint_step_008193.jld2")
@info "Loading" ic_path

Nλ, Nφ, Nz, ρ_data, ρu_data, ρv_data, ρθ_data, ρqv_data = JLD2.jldopen(ic_path, "r") do f
    (f["Nλ"], f["Nφ"], f["Nz"], f["ρ"], f["ρu"], f["ρv"], f["ρθ"], f["ρqᵛ"])
end

grid = LatitudeLongitudeGrid(CPU();
    size = (Nλ, Nφ, Nz),
    halo = (4, 4, 4),
    latitude = (-80, 80),
    longitude = (0, 360),
    z = (0, 30e3))

FT = Float32
safe_ρ = max.(ρ_data, FT(0.001))
θ_data = ρθ_data ./ safe_ρ
qv_data = ρqv_data ./ safe_ρ .* FT(1000)
u_data = ρu_data ./ safe_ρ
v_data = ρv_data[:, 1:Nφ, :] ./ safe_ρ
wspd_data = sqrt.(u_data.^2 .+ v_data.^2)

θ_field = CenterField(grid)
qv_field = CenterField(grid)
wspd_field = CenterField(grid)

Oceananigans.interior(θ_field) .= FT.(θ_data)
Oceananigans.interior(qv_field) .= FT.(qv_data)
Oceananigans.interior(wspd_field) .= FT.(wspd_data)

@info "Rendering..."

k_sfc = 1
elev = 0.4
azim = 1.2

fig = Figure(size = (1800, 700))

# θ — slightly less clipping than before (was 260–310, now 250–315)
ax1 = Axis3(fig[1, 1]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax1); hidespines!(ax1)
plt1 = surface!(ax1, view(θ_field, :, :, k_sfc); colormap = Reverse(:RdYlBu), colorrange = (250, 315))
Colorbar(fig[2, 1], plt1; vertical = false, width = Relative(0.6), label = "θ [K]")

# qv — Reverse(:dense) so high moisture is dark/vivid, low is light
ax2 = Axis3(fig[1, 2]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax2); hidespines!(ax2)
plt2 = surface!(ax2, view(qv_field, :, :, k_sfc); colormap = :tempo, colorrange = (0, 34))
Colorbar(fig[2, 2], plt2; vertical = false, width = Relative(0.6), label = "qv [g/kg]")

# Wind speed
ax3 = Axis3(fig[1, 3]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax3); hidespines!(ax3)
plt3 = surface!(ax3, view(wspd_field, :, :, k_sfc); colormap = :solar, colorrange = (0, 30))
Colorbar(fig[2, 3], plt3; vertical = false, width = Relative(0.6), label = "Wind speed [m/s]")

Label(fig[0, :], "1/8° checkpoint — surface fields on the sphere", fontsize = 20)

colgap!(fig.layout, -100)
rowgap!(fig.layout, 1, -50)
rowsize!(fig.layout, 1, Relative(0.8))

outpath = joinpath(@__DIR__, "sphere_eighth_degree.png")
save(outpath, fig; px_per_unit = 4)
@info "Saved" outpath
