using GordonBell25
using Oceananigans
using Reactant

if !GordonBell25.is_distributed_env_present()
    using MPI
    MPI.Init()
end

throw_error = true
include_halos = true
rtol = sqrt(eps(Float64))
atol = 0

GordonBell25.initialize(; single_gpu_per_process=false)
@show Ndev = length(Reactant.devices())

Rx, Ry = GordonBell25.factors(Ndev)

rarch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition = Partition(Rx, Ry, 1)
)

rank = Reactant.Distributed.local_rank()

H = 8
Tx = 64 * Rx
Ty = 64 * Ry
Nz = 16

Nx = Tx - 2H
Ny = Ty - 2H

model_kw = (
    halo = (H, H, H),
    Δt = 1e-9,
)

varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel
@assert rmodel.architecture isa Distributed

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)

@info "At the beginning:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@jit Oceananigans.initialize!(rmodel)
Oceananigans.initialize!(vmodel)



using InteractiveUtils

using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans.Architectures: architecture
using Oceananigans: fields, prognostic_fields, boundary_conditions, instantiated_location
using Oceananigans.Fields: tupled_fill_halo_regions!, fill_reduced_field_halos!, default_indices, data, ReducedField, FullField
using Oceananigans.BoundaryConditions: fill_halo_regions!, permute_boundary_conditions, fill_halo_event!,
                                       extract_bc, extract_west_bc, extract_south_bc, extract_bottom_bc,
                                       fill_west_and_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!,
                                       fill_halo_size, fill_halo_offset, parent_size_and_offset, fill_periodic_west_and_east_halo!


"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function my_fill_halo_regions!(c, grid;
                            fill_boundary_normal_velocities = true, kwargs...)
    arch = architecture(grid)

    my_fill_halo_event!(c, arch, grid; kwargs...)

    return nothing
end

function my_fill_halo_event!(c, arch, grid;
                          async = false, # This kwargs is specific to DistributedGrids, here is does nothing
                          kwargs...)

    # Calculate size and offset of the fill_halo kernel
    # We assume that the kernel size is the same for west and east boundaries,
    # south and north boundaries, and bottom and top boundaries
    size   = :yz
    offset = (0, 0)

    my_fill_west_and_east_halo!(c, offset, arch, grid; kwargs...)

    return nothing
end

function my_fill_west_and_east_halo!(c, offset, arch, grid; only_local_halos = false, kw...)

    c_parent = parent.(c)
    yz_size  = (minimum([size(t, 2) for t in c_parent]), minimum([size(t, 3) for t in c_parent]))
    offset   = (0, 0)

    launch!(arch, grid, KernelParameters(yz_size, offset), fill_periodic_west_and_east_halo!, c_parent, Val(grid.Hx), grid.Nx; kw...)
    return nothing
end

@inline my_prognostic_fields(model) = (model.velocities.u.data, model.velocities.v.data, model.tracers.T.data, model.tracers.S.data)

my_fill_halo_regions!(my_prognostic_fields(vmodel), vmodel.grid, async=true)
@jit my_fill_halo_regions!(my_prognostic_fields(rmodel), rmodel.grid, async=true)

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
