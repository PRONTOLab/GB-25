cᵢ(x, y, z) = exp(-(x^2 + y^2) / 128)

function tracer_only_model(arch; Nx, Ny, Nz, Δt,
    tracer_advection=Centered(order=2))

    grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7),
                           x=(-Nx/2, Nx/2), y=(-Ny/2, Ny/2), z=(0, Nz),
                           topology = (Periodic, Periodic, Bounded))

    model = HydrostaticFreeSurfaceModel(;
        grid, velocities=PrescribedVelocityFields(),
        tracers=:c, tracer_advection,
    )

    Random.seed!(42)
    set!(model, c=cᵢ)
    model.clock.last_Δt = Δt

    return model
end
