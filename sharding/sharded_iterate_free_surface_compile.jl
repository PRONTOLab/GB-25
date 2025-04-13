using Dates
using GordonBell25
using Reactant
using KernelAbstractions: @kernel, @index
using Oceananigans
using Printf

# This must be called before `GordonBell25.initialize`!
GordonBell25.preamble(; rendezvous_warn=20, rendezvous_terminate=40)
@show Ndev = length(Reactant.devices())
local_arch = Oceananigans.ReactantState()
#local_arch = CPU()

if Ndev == 1
    Rx = Ry = 1
    rank = 0
    arch = local_arch
else
    Rx, Ry = GordonBell25.factors(Ndev)
    arch = Oceananigans.Distributed(
        local_arch,
        partition = Partition(Rx, Ry, 1)
    )

    rank = arch.local_rank
end

@show arch

H = 8
Nx = 48 * Rx
Ny = 24 * Ry
Nz = 4

@info "[$rank] Generating model..." now(UTC)
model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)
@show model

#=
free_surface_func!(model) = Oceananigans.Models.HydrostaticFreeSurfaceModels.step_free_surface!(
    model.free_surface,
    model,
    model.timestepper,
    model.clock.last_Δt)
=#

function free_surface_func!(model)
    substepping = model.free_surface.substepping
    Ns = Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces.calculate_substeps(substepping, model.clock.last_Δt)
    δ, w = Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces.calculate_adaptive_settings(substepping, Ns)
    Δτ = δ * model.clock.last_Δt
    Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces.iterate_split_explicit!(
        model.free_surface,
        model.free_surface.η.grid,
        model.timestepper.Gⁿ.U,
        model.timestepper.Gⁿ.V,
        Δτ,
        w,
        Val(Ns)
    )

    return nothing
end
    
function myloop!(model, Nt)
    @trace track_numbers=false for _ = 1:Nt
        free_surface_func!(model)
    end
    return nothing
end

if local_arch isa CPU
    step_free_surface!(model)
    error("done")
end

name = "iterate_free_surface"
xla_code = @code_xla raise=true free_surface_func!(model)
open("sharded_$(name)_$Ndev.xla.txt", "w") do io
    print(io, xla_code)
end

xla_loop_code = @code_xla raise=true myloop!(model, ConcreteRNumber(3))
open("sharded_$(name)_loop_$Ndev.xla.txt", "w") do io
    print(io, xla_loop_code)
end
