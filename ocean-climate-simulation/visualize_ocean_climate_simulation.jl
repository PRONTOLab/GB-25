using Oceananigans
using GLMakie

filename = "ocean_climate_simulation_surface.jld2"
Tt = FieldTimeSeries(filename, "T")
ut = FieldTimeSeries(filename, "u")

Nt = length(Tt)
Nz = size(grid, 3)
n = Observable(Nt)
Tn = @lift interior(Tt[$n], :, :, 1)
un = @lift interior(ut[$n], :, :, 1)

fig = Figure(size=(800, 800))
axT = Axis(fig[1, 1])
axu = Axis(fig[2, 1])

hm = heatmap!(axT, Tn, colormap=:thermal, colorrange=(-1, 30))
Colorbar(fig[1, 2], hm, label="Temperature (ᵒC)")

hm = heatmap!(axu, un, colormap=:balance, colorrange=(-0.5, 0.5))
Colorbar(fig[2, 2], hm, label="Zonal velocity (m s⁻¹)")

display(fig)

