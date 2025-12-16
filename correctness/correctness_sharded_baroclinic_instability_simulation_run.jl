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
using Oceananigans.Fields: tupled_fill_halo_regions!, fill_reduced_field_halos!, default_indices, data
using Oceananigans.BoundaryConditions: fill_halo_regions!, permute_boundary_conditions, fill_halo_event!,
                                       extract_bc, extract_west_bc, extract_south_bc, extract_bottom_bc,
                                       fill_west_and_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!,
                                       fill_halo_size, fill_halo_offset, parent_size_and_offset, fill_periodic_west_and_east_halo!


@show @which prognostic_fields(rmodel)
@show @which prognostic_fields(vmodel)

@show @which fields(rmodel)
@show @which fields(vmodel)

function my_tupled_fill_halo_regions!(fields, grid, args...; kwargs...)

    not_reduced_fields = fill_reduced_field_halos!(fields, args...; kwargs)

    @show @which fill_reduced_field_halos!(fields, args...; kwargs)
    
    if !isempty(not_reduced_fields) # ie not reduced, and with default_indices

        my_fill_halo_regions!(map(data, not_reduced_fields),
                           map(boundary_conditions, not_reduced_fields),
                           default_indices(3),
                           map(instantiated_location, not_reduced_fields),
                           grid, args...; kwargs...)
    end
    
    return nothing
end

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function my_fill_halo_regions!(c, boundary_conditions, indices, loc, grid, args...;
                            fill_boundary_normal_velocities = true, kwargs...)
    arch = architecture(grid)

    fill_halos! = fill_west_and_east_halo!
    sides       = [:west_and_east]
    bcs         = extract_bc(boundary_conditions, Val(sides[1]))

    my_fill_halo_event!(c, bcs, indices, loc, arch, grid, args...; kwargs...)

    return nothing
end

function my_fill_halo_event!(c, bcs, indices, loc, arch, grid, args...;
                          async = false, # This kwargs is specific to DistributedGrids, here is does nothing
                          kwargs...)

    # Calculate size and offset of the fill_halo kernel
    # We assume that the kernel size is the same for west and east boundaries,
    # south and north boundaries, and bottom and top boundaries
    size   = :yz
    offset = (0, 0)

    @show @which fill_west_and_east_halo!(c, bcs..., size, offset, loc, arch, grid, args...; kwargs...)

    my_fill_west_and_east_halo!(c, size, offset, loc, arch, grid, args...; kwargs...)

    return nothing
end

function my_fill_west_and_east_halo!(c, size, offset, loc, arch, grid, args...; only_local_halos = false, kw...)
    c_parent, yz_size, offset = parent_size_and_offset(c, 2, 3, size, offset)
    launch!(arch, grid, KernelParameters(yz_size, offset), fill_periodic_west_and_east_halo!, c_parent, Val(grid.Hx), grid.Nx; kw...)
    return nothing
end

my_tupled_fill_halo_regions!(prognostic_fields(vmodel), vmodel.grid, vmodel.clock, fields(vmodel), async=true)
@jit my_tupled_fill_halo_regions!(prognostic_fields(rmodel), rmodel.grid, rmodel.clock, fields(rmodel), async=true)

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

#=

GordonBell25.sync_states!(rmodel, vmodel)
rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
@showtime rfirst!(rmodel)
@showtime GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

rstep! = @compile sync=true raise=true GordonBell25.time_step!(rmodel)

@info "Warm up:"
@showtime rstep!(rmodel)
@showtime rstep!(rmodel)
@showtime GordonBell25.time_step!(vmodel)
@showtime GordonBell25.time_step!(vmodel)

Nt = 10
@info "Time step with Reactant:"
for _ in 1:Nt
    @showtime rstep!(rmodel)
end

@info "Time step vanilla:"
for _ in 1:Nt
    @showtime GordonBell25.time_step!(vmodel)
end

@info "After $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)
@jit Oceananigans.TimeSteppers.update_state!(rmodel)

@info "After syncing and updating state again:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true GordonBell25.loop!(rmodel, rNt)
@showtime rloop!(rmodel, rNt)
@showtime GordonBell25.loop!(vmodel, Nt)

@info "After a loop of $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
=#