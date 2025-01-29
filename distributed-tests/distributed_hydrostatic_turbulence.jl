using Oceananigans
using Printf
using MPI

# Run with 
#
# ```julia 
#   mpiexec -n 4 julia --project distributed_hydrostatic_turbulence.jl
# ```

MPI.Init()
arch = Distributed(CPU())
#arch = CPU()
Nx = Ny = 128

grid = RectilinearGrid(arch; size=(Nx, Ny, 4), extent=(4π, 4π, 1), halo=(8, 8, 8))

free_surface = SplitExplicitFreeSurface(grid, gravitational_acceleration=4, substeps=16)
model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO())

ui(x, y, z) = randn() + 0.1 * sin(x) * sin(y)
set!(model, u=ui, v=ui)

Δx = minimum_xspacing(grid)
Δt = 0.1 * Δx
simulation = Simulation(model; Δt, stop_iteration=400)

function progress(sim)
    u, v, w = sim.model.velocities
    max_u = maximum(u)
    @info @sprintf("Iter: %d, time: %.2f, max(u): %.1e",
                   iteration(sim), time(sim), max_u)
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

run!(simulation)

# Quick visualization
if !(arch isa Distributed)
    using GLMakie
    u, v, w = model.velocities
    ζ = Field(∂x(v) - ∂y(u))
    heatmap(view(ζ, :, :, 1), axis=(; aspect=1))
end

