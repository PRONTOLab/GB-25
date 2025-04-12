using Dates
using GordonBell25
using Reactant
using KernelAbstractions: @kernel, @index
using Oceananigans
using Printf

# This must be called before `GordonBell25.initialize`!
GordonBell25.preamble(; rendezvous_warn=20, rendezvous_terminate=40)
@show Ndev = length(Reactant.devices())

Rx, Ry = GordonBell25.factors(Ndev)
arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition = Partition(Rx, Ry, 1)
)

rank = arch.local_rank

#=
Rx = Ry = 1
rank = 0
arch = CPU()
=#

@show arch

H = 8
Nx = 48 * Rx
Ny = 24 * Ry
Nz = 4

@info "[$rank] Generating model..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@show model

function simple_ab2_step!(model, Δt)
    grid = model.grid
    FT = eltype(grid)
    χ = convert(FT, model.timestepper.χ)
    Δt = convert(FT, Δt)
    # ab2_step_velocities!(model.velocities, model, Δt, χ)
    return nothing
end

@kernel function ab2_step_field!(u, Δt, χ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    FT = eltype(u)
    Δt = convert(FT, Δt)
    one_point_five = convert(FT, 1.5)
    oh_point_five  = convert(FT, 0.5)
    not_euler = χ != convert(FT, -0.5) # use to prevent corruption by leftover NaNs in G⁻

    @inbounds begin
        Gu = (one_point_five + χ) * Gⁿ[i, j, k] - (oh_point_five + χ) * G⁻[i, j, k] * not_euler
        u[i, j, k] += Δt * Gu
    end
end

function ab2_step_velocities!(velocities, model, Δt, χ)
    Gⁿ = model.timestepper.Gⁿ.u
    G⁻ = model.timestepper.G⁻.u
    u = model.velocities.u

    Oceananigans.Utils.launch!(model.architecture, model.grid, :xyz, ab2_step_field!, u, Δt, χ, Gⁿ, G⁻)

    # implicit_step!(u,
    #                model.timestepper.implicit_solver,
    #                model.closure,
    #                model.diffusivity_fields,
    #                nothing,
    #                model.clock, 
    #                Δt)

    return nothing
end

@jit first_time_step!(model)

# first_time_step_xla    = @code_xla raise=true GordonBell25.first_time_step!(model)
# time_step_xla          = @code_xla raise=true GordonBell25.time_step!(model)
# compute_tendencies_xla = @code_xla raise=true Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_tendencies!(model, [])
# update_state_xla       = @code_xla raise=true Oceananigans.Models.HydrostaticFreeSurfaceModels.update_state!(model)

ab2_step_xla = @code_xla raise=true simple_ab2_step!(model, model.clock.last_Δt)
#simple_ab2_step!(model, model.clock.last_Δt)

codes = [
    # ("first_time_step", first_time_step_xla),
    # ("time_step", time_step_xla),
    ("simple_ab2_step", ab2_step_xla),
    # ("compute_tendencies", compute_tendencies_xla),
    # ("update_state", update_state_xla)
]

for pair in codes
    name, code = pair             
    open("sharded_$(name)_$Ndev.xla", "w") do io
        print(io, code)
    end
end
