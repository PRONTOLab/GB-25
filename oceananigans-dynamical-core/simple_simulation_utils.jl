# Some simple utilities for adding output and visualizing the results of a simulation

"""
    add_simple_output!(simulation, filename, schedule; indices=(:, :, Nz))

"""
function add_simple_output!(simulation, filename, schedule;
                            indices=(:, :, size(simulation.model.grid, 3)))

    u, v, w = simulation.model.velocities
    ζ = ∂x(v) - ∂y(u)
    s = @at (Center, Center, Center) sqrt(u^2 + v^2)

    tracers = simulation.model.tracers
    outputs = merge((; u, v, w, ζ, s), tracers)

    output_writer = JLD2OutputWriter(simulation.model, outputs; filename, schedule, indices,
                                     overwrite_existing = true)

    simulation.output_writers[:jld2] = output_writer

    return nothing
end

