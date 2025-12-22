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


using InteractiveUtils

using KernelAbstractions
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using OffsetArrays

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

using Oceananigans.Utils: KernelParameters, launch!, _launch!, configure_kernel, OffsetStaticSize, interior_work_layout, work_layout, mapped_kernel
using Oceananigans.Architectures: architecture, child_architecture
using Oceananigans: fields, prognostic_fields, boundary_conditions, instantiated_location
using Oceananigans.Fields: tupled_fill_halo_regions!, fill_reduced_field_halos!, default_indices, data, ReducedField, FullField
using Oceananigans.BoundaryConditions: fill_halo_regions!, permute_boundary_conditions, fill_halo_event!,
                                       extract_bc, extract_west_bc, extract_south_bc, extract_bottom_bc,
                                       fill_west_and_east_halo!, fill_south_and_north_halo!, fill_bottom_and_top_halo!,
                                       fill_halo_size, fill_halo_offset, parent_size_and_offset, fill_periodic_west_and_east_halo!


function my_fill_west_and_east_halo!(c, dev, Hx, Nx)

    c_parent = parent.(c)

    _my_launch!(dev, my_fill_periodic_west_and_east_halo!, c_parent, Val(Hx), Nx)
    return nothing
end

@inline function _my_launch!(dev, kernel!, first_kernel_arg, other_kernel_args...)

    workgroup = KernelAbstractions.NDIteration.StaticSize{(16, 16)}()
    worksize = OffsetStaticSize{(1:128, 1:32)}()

    loop! = kernel!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(first_kernel_arg, other_kernel_args...)

    return nothing
end

@kernel function my_fill_periodic_west_and_east_halo!(c, ::Val{H}, N) where {H}
    j, k = @index(Global, NTuple)
    ntuple(Val(length(c))) do n
        Base.@_inline_meta
        @unroll for i = 1:H
            @inbounds begin
                  c[n][i, j, k]     = c[n][N+i, j, k] # west
                  c[n][N+H+i, j, k] = c[n][H+i, j, k] # east
            end
        end
    end
end

Hx = vmodel.grid.Hx

rdev = ReactantKernelAbstractionsExt.ReactantBackend()
vdev = KernelAbstractions.CPU()

@show typeof(vmodel.velocities.u.data), size(vmodel.velocities.u.data)
@show typeof(rmodel.velocities.u.data), size(rmodel.velocities.u.data)

u = zeros(128, 128, 32)
vu = OffsetArray(u, -7:120, -7:120, -7:24)
ru = Reactant.to_rarray(vu)

@inline my_prognostic_fields(model) = (model.velocities.u.data,)

vfields = (vu,)
rfields = (ru,)

@show typeof(vu), size(vu)
@show typeof(ru), size(ru)

opts = Reactant.CompileOptions(; max_constant_threshold=1000, raise=true)
#@show @code_hlo compile_options = opts my_fill_west_and_east_halo!(my_prognostic_fields(rmodel), rdev, Hx, Nx)

my_fill_west_and_east_halo!(my_prognostic_fields(vmodel), vdev, Hx, Nx)
@jit my_fill_west_and_east_halo!(my_prognostic_fields(rmodel), rdev, Hx, Nx)

