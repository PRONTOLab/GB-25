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

@show @which fields(rmodel)
@show @which fields(vmodel)

function my_tupled_fill_halo_regions!(fields, grid, args...; kwargs...)

    not_reduced_fields = my_fill_reduced_field_halos!(fields, args...; kwargs)
    
    if !isempty(not_reduced_fields) # ie not reduced, and with default_indices

        my_fill_halo_regions!(map(data, not_reduced_fields),
                           map(instantiated_location, not_reduced_fields),
                           grid, args...; kwargs...)
    end
    
    return nothing
end

function my_fill_reduced_field_halos!(fields, args...; kwargs)

    @show fields

    not_reduced_fields = Field[]
    for f in fields
        bcs = boundary_conditions(f)
        if !isnothing(bcs)
            if f isa ReducedField || !(f isa FullField)
                # Windowed and reduced fields
                #fill_halo_regions!(f, args...; kwargs...)
                @info "We got a ReducedField"
            else
                push!(not_reduced_fields, f)
                @info "We did NOT get a ReducedField"
            end
        end
    end

    return tuple(not_reduced_fields...)
end

"Fill halo regions in ``x``, ``y``, and ``z`` for a given field's data."
function my_fill_halo_regions!(c, loc, grid, args...;
                            fill_boundary_normal_velocities = true, kwargs...)
    arch = architecture(grid)

    my_fill_halo_event!(c, loc, arch, grid, args...; kwargs...)

    return nothing
end

function my_fill_halo_event!(c, loc, arch, grid, args...;
                          async = false, # This kwargs is specific to DistributedGrids, here is does nothing
                          kwargs...)

    # Calculate size and offset of the fill_halo kernel
    # We assume that the kernel size is the same for west and east boundaries,
    # south and north boundaries, and bottom and top boundaries
    size   = :yz
    offset = (0, 0)

    my_fill_west_and_east_halo!(c, offset, loc, arch, grid, args...; kwargs...)

    return nothing
end

function my_fill_west_and_east_halo!(c, offset, loc, arch, grid, args...; only_local_halos = false, kw...)

    c_parent = parent.(c)
    yz_size  = (minimum([size(t, 2) for t in c_parent]), minimum([size(t, 3) for t in c_parent]))
    offset   = (0, 0)

    launch!(arch, grid, KernelParameters(yz_size, offset), fill_periodic_west_and_east_halo!, c_parent, Val(grid.Hx), grid.Nx; kw...)
    return nothing
end

@inline my_prognostic_fields(model) = (u=model.velocities.u, v=model.velocities.v, T=model.tracers.T, S=model.tracers.S)

my_tupled_fill_halo_regions!(my_prognostic_fields(vmodel), vmodel.grid, vmodel.clock, fields(vmodel), async=true)
@jit my_tupled_fill_halo_regions!(my_prognostic_fields(rmodel), rmodel.grid, rmodel.clock, fields(rmodel), async=true)

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