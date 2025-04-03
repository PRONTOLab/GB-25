cᵢ(x, y, z) = exp(-(x^2 + y^2) / 128)

@inline function func(i, j, k, grid, clock, fields)
    c = @inbounds fields.c[i, j, k]
    return c - c^3 + log(abs(c) + 1)
end

function tracer_only_model(arch; Nx, Ny, Nz, Δt)

    grid = LatitudeLongitudeGrid(arch,
        size = (Nx, Ny, Nz),
        halo = (1, 1, 1),
        z = (0, 1),
        latitude = (-80, 80),
        longitude = (0, 360)
    )

    boundary_conditions = FieldBoundaryConditions(
        top = nothing,
        bottom = nothing,
        east = nothing,
        west = nothing,
        north = nothing,
        south = nothing,
    )

    c = CenterField(grid; boundary_conditions)
    forcing = Forcing(func, discrete_form=true)

    model = HydrostaticFreeSurfaceModel(; grid,
        velocities = PrescribedVelocityFields(),
        free_surface = nothing,
        tracers  = (; c),
        forcing = (; c=forcing),
        tracer_advection = nothing,
    )

    Random.seed!(42)
    c = model.tracers.c
    cᵢ = rand(size(parent(c))...)
    copyto!(parent(c), cᵢ)
    model.clock.last_Δt = Δt

    return model
end

