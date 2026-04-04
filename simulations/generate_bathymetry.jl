using Oceananigans
using Oceananigans.Units
using NumericalEarth
using JLD2
using CairoMakie

# Generate bathymetry at 1/12 degree and save to disk.
# Run scripts then interpolate from this cached field to their target grid.

resolution = 1/12 # degrees — source resolution for caching

Nx = convert(Int, 384 / resolution) # 4608
Ny = convert(Int, 192 / resolution) # 2304
Nz = 1 # only need horizontal

# Use a simple grid for regridding (z doesn't matter for bathymetry)
grid = LatitudeLongitudeGrid(CPU(); size=(Nx, Ny, Nz), halo=(8, 8, 8),
                             z = (-100, 0),
                             latitude = (-80, 80),
                             longitude = (0, 360))

@info "Regridding ETOPO bathymetry to $(resolution)° ($Nx × $Ny)..."
bottom_height = regrid_bathymetry(grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 5,
                                  major_basins = 1)

# Save the interior data as a plain array
filename = joinpath(@__DIR__, "..", "bathymetry_twelfth_degree.jld2")
h = Array(interior(bottom_height, :, :, 1))

@info "Saving to $filename..."
jldsave(filename;
        bottom_height = h,
        Nx = Nx,
        Ny = Ny,
        latitude = (-80, 80),
        longitude = (0, 360))

@info "Done! Bathymetry saved: $Nx × $Ny at $(resolution)°"

# ============================================================
# Visualize
# ============================================================

λ = range(resolution/2, 360 - resolution/2, length=Nx)
φ = range(-80 + 160/Ny/2, 80 - 160/Ny/2, length=Ny)

fig = Figure(size = (1000, 500))
ax = Axis(fig[1, 1],
          title = "ETOPO bathymetry at $(resolution)° ($Nx × $Ny)",
          xlabel = "Longitude",
          ylabel = "Latitude")

hm = heatmap!(ax, collect(λ), collect(φ), h, colormap = :deep, colorrange = (-6000, 0))
Colorbar(fig[1, 2], hm, label = "Bottom height (m)")

save(joinpath(@__DIR__, "..", "bathymetry_twelfth_degree.png"), fig)
@info "Saved bathymetry_twelfth_degree.png"
