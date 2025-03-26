function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture, resolution, Nz)
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 180 / resolution)

    # Time step. This must be decreased as resolution is decreased.
    Δt = 1minutes

    # Grid setup
    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    #underlying_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces,
    #                                        longitude=(0, 360), latitude=(-80, 80))
    zb = z_faces[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

    return @gbprofile "ImmersedBoundaryGrid" ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands);
                                                                  active_cells_map=false)
end

function simple_latitude_longitude_grid(arch, resolution, Nz)
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 160 / resolution)

    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces,
        latitude = (-80, 80),
        longitude = (0, 360)
    )

    return grid
end

