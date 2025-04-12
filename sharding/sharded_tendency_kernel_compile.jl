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

function compute_Gu!(model)
    active_cells_map = nothing
    grid = model.grid
    arch = grid.architecture
    u_immersed_bc = nothing
    u_forcing = model.forcing.u
    kernel_parameters = Oceananigans.Models.HydrostaticFreeSurfaceModels.interior_tendency_kernel_parameters(arch, grid)

    start_momentum_kernel_args = (model.advection.momentum,
                                  model.coriolis,
                                  model.closure)

    end_momentum_kernel_args = (model.velocities,
                                model.free_surface,
                                model.tracers,
                                model.buoyancy,
                                model.diffusivity_fields,
                                model.pressure.pHY′,
                                model.auxiliary_fields,
                                model.vertical_coordinate,
                                model.clock)

    u_kernel_args = tuple(start_momentum_kernel_args..., u_immersed_bc, end_momentum_kernel_args..., u_forcing)

    Oceananigans.Utils.launch!(
        arch,
        grid,
        kernel_parameters,
        Oceananigans.Models.HydrostaticFreeSurfaceModels.compute_hydrostatic_free_surface_Gu!,
        model.timestepper.Gⁿ.u,
        grid, 
        active_cells_map,
        u_kernel_args;
        active_cells_map)

    return nothing
end

function myloop!(model, Nt)
    @trace track_numbers=false for _ = 1:Nt
        compute_Gu!(model)
    end
    return nothing
end

if local_arch isa CPU
    compute_Gu!(model)
    error("done")
end

compute_Gu_xla = @code_xla raise=true compute_Gu!(model)
open("sharded_compute_Gu_$Ndev.xla", "w") do io
    print(io, compute_Gu_xla)
end

compute_Gu_loop_xla = @code_xla raise=true myloop!(model, ConcreteRNumber(3))
open("sharded_compute_Gu_loop_$Ndev.xla", "w") do io
    print(io, compute_Gu_loop_xla)
end
