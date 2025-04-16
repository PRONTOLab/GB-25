using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using GLMakie

function geographic2cartesian(λ, φ, r=1)
    # Nλ = length(λ)
    # Nφ = length(φ)
    # λ = repeat(reshape(λ, Nλ, 1), 1, Nφ)
    # φ = repeat(reshape(φ, 1, Nφ), Nλ, 1)

    λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
    φ_azimuthal = 90 .- φ   # Convert to φ ∈ [0°, 180°] (0° at north pole)

    x = @. r * cosd(λ_azimuthal) * sind(φ_azimuthal)
    y = @. r * sind(λ_azimuthal) * sind(φ_azimuthal)
    z = @. r * cosd(φ_azimuthal)

    return x, y, z
end

filename = "deep_half_degree_simulation_io_surface.jld2"
ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")
ζt = FieldTimeSeries(filename, "ζ")

grid = st.grid
λ = λnodes(grid, Center(), Center(), Center())
φ = φnodes(grid, Center(), Center(), Center())
x, y, z = geographic2cartesian(λ, φ)

Nx, Ny, Nz = size(grid)

#=
land = interior(ζt.grid.immersed_boundary.bottom_height) .== 0
Nt = length(st)
for n in 1:Nt
    interior(ζt[n])[land] .= NaN
    interior(st[n])[land] .= NaN
end
=#

n = Observable(1)
sn = @lift interior(st[$n], :, :, 1)
ζn = @lift interior(ζt[$n], :, :, 1)

fig = Figure(size=(1500, 1900))

kw = (elevation=0.7, azimuth=2.1, aspect=:equal)
axζ = Axis3(fig[1, 1]; kw...)
axs = Axis3(fig[1, 2]; kw...)
surface!(axζ, x, y, z, color=ζn, colormap=:haline, nan_color=:lightgray)
surface!(axs, x, y, z, color=sn, colorrange=(0, 1.0), colormap=:magma, nan_color=:lightgray)

t = st.times
title = @lift string(prettytime(t[$n]), " after Jan 1 1993")
Label(fig[0, 1:2], title, fontsize=24)

hidedecorations!(axζ)
hidedecorations!(axs)
hidespines!(axζ)
hidespines!(axs)

rowgap!(fig.layout, 1, Relative(-0.4))
colgap!(fig.layout, 1, Relative(-0.1))

display(fig)

#record(fig, "quarter_degree_thermo_sea_ice.mp4", 1:Nt, framerate=12) do nn
record(fig, "half_degree_thermo_sea_ice.mp4", 1:Nt, framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end

