using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
    calculate_substeps,
    calculate_adaptive_settings

function simple_model(arch)

    Nx = Ny = Nz = 16
    grid = LatitudeLongitudeGrid(arch,
                                 topology = (Periodic, Bounded, Bounded),
                                 size = (Nx, Ny, Nz),
                                 longitude = (-10, 10),
                                 latitude = (40, 60),
                                 z = (-1000, 0),
                                 halo = (6, 6, 6))

    free_surface = SplitExplicitFreeSurface(substeps=10)
    model = HydrostaticFreeSurfaceModel(; grid, free_surface)
    model.clock.last_Δt = 60

    return model
end

c_model = simple_model(CPU())
r_model = simple_model(Oceananigans.Architectures.ReactantState())

using Random
Random.seed!(123)
pηc = parent(c_model.free_surface.η)
pηr = parent(r_model.free_surface.η)
η₀ = rand(size(pηc)...)
copyto!(pηc, η₀)
copyto!(pηr, Reactant.to_rarray(η₀))

function launch_problem_kernel!(model)
    free_surface = model.free_surface
    η           = free_surface.η
    grid        = free_surface.η.grid
    arch        = Oceananigans.Architectures.architecture(grid)
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    parameters  = free_surface.kernel_parameters

    Nsubsteps = calculate_substeps(model.free_surface.substepping, model.clock.last_Δt)
    fractional_Δt, weights = calculate_adaptive_settings(model.free_surface.substepping, Nsubsteps)
    Δτᴮ = fractional_Δt * c_model.clock.last_Δt
    val_substeps = Val(1)
    GUⁿ = model.timestepper.Gⁿ.U
    GVⁿ = model.timestepper.Gⁿ.V

    # unpack state quantities, parameters and forcing terms
    U, V    = free_surface.barotropic_velocities
    η̅, U̅, V̅ = state.η, state.U, state.V

    barotropic_velocity_kernel!, _ = Oceananigans.Utils.configure_kernel(
        arch,
        grid,
        parameters,
        Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces._split_explicit_barotropic_velocity!
    )

    averaging_weight = weights[1]
    barotropic_velocity_kernel!(averaging_weight, grid, Δτᴮ, η, U, V, η̅, U̅, V̅, GUⁿ, GVⁿ, g, timestepper)

    return nothing
end

r_problem! = @compile sync=true raise=true launch_problem_kernel!(r_model)
@time r_problem!(r_model)
@time launch_problem_kernel!(c_model)

function compare(c, r, name="")
    pc = Array(parent(c))
    pr = Array(parent(r))
    @show name, maximum(pc .- pr)
end

Uc = c_model.free_surface.barotropic_velocities.U
Ur = r_model.free_surface.barotropic_velocities.U
compare(Uc, Ur, "U")

