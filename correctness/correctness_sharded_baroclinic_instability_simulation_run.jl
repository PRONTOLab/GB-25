using NPZ
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
    coriolis = nothing,
    # buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState()),
    momentum_advection = nothing,
    tracer_advection = nothing,
)

varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel
@assert rmodel.architecture isa Distributed

exported_jax_file_dir = joinpath(@__DIR__, "exported_jax_files")

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)

@info "At the beginning:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Reactant.Serialization.export_to_enzymejax(
    Oceananigans.Models.NonhydrostaticModels.update_hydrostatic_pressure!,
    rmodel.pressure.pHY′, rmodel.architecture, rmodel.grid, rmodel.buoyancy, rmodel.tracers;
    output_dir=exported_jax_file_dir,
    compile_options=CompileOptions(raise=true),
)

Oceananigans.Models.NonhydrostaticModels.update_hydrostatic_pressure!(
    vmodel.pressure.pHY′, vmodel.architecture, vmodel.grid, vmodel.buoyancy, vmodel.tracers)
@jit Oceananigans.Models.NonhydrostaticModels.update_hydrostatic_pressure!(
    rmodel.pressure.pHY′, rmodel.architecture, rmodel.grid, rmodel.buoyancy, rmodel.tracers)

@info "After updating hydrostatic pressure:"
GordonBell25.compare_interior("pHY′", rmodel.pressure.pHY′, vmodel.pressure.pHY′)
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

#@jit Oceananigans.initialize!(rmodel)
#Oceananigans.initialize!(vmodel)

using InteractiveUtils
using KernelAbstractions
using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: Δrᶜᶜᶜ, Δrᶜᶜᶠ, Δrᶜᶠᶜ, Δrᶜᶠᶠ, Δrᶠᶜᶜ, Δrᶠᶜᶠ, Δrᶠᶠᶜ, Δrᶠᶠᶠ
using Oceananigans.Utils: launch!, _launch!, KernelParameters, configure_kernel, interior_work_layout, work_layout, mapped_kernel
using Oceananigans.BoundaryConditions: BoundaryCondition, getbc, Flux, Open, _fill_south_and_north_halo!, _fill_south_halo!, _fill_north_halo!, _fill_flux_north_halo!
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.Grids: get_active_column_map, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, column_depthᶠᶜᵃ, column_depthᶜᶠᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: _compute_barotropic_mode!

using OffsetArrays

const ReactantKernelAbstractionsExt = Base.get_extension(
    Reactant, :ReactantKernelAbstractionsExt
)

rdev = ReactantKernelAbstractionsExt.ReactantBackend()
vdev = KernelAbstractions.CPU()


u   = ones(Float64, 128, 128, 32)
vu  = OffsetArray(u, -7:120, -7:120, -7:24)
vu .= vmodel.velocities.u.data
ru  = deepcopy(rmodel.velocities.u.data) #Reactant.to_rarray(vu) # This doesn't work


vu2  = ones(Float64, 128, 128, 32)
vu2 .= OffsetArrays.no_offset_view(vmodel.velocities.u.data)
ru2  = Reactant.to_rarray(vu2)
vu2  = OffsetArray(vu2, -7:120, -7:120, -7:24)
ru2  = OffsetArray(ru2, -7:120, -7:120, -7:24)

ru3 = Reactant.to_rarray(vu)

U  = zeros(Float64, 128, 128, 1)
vU = OffsetArray(U, -7:120, -7:120, 1:1)
rU = Reactant.to_rarray(vU)
rU2 = Reactant.to_rarray(vU)
rU3 = Reactant.to_rarray(vU)

@info "Differences between ru, ru2, and ru3:"
@show maximum(abs.(convert(Array, parent(ru)) - convert(Array, parent(rmodel.velocities.u.data))))
@show maximum(abs.(convert(Array, parent(ru2)) - convert(Array, parent(rmodel.velocities.u.data))))
@show maximum(abs.(convert(Array, parent(ru3)) - convert(Array, parent(rmodel.velocities.u.data))))

@show typeof(rmodel.velocities.u.data)
@show typeof(ru)
@show typeof(ru2)
@show typeof(ru3)

@show size(ru)
@show size(rmodel.velocities.u.data)

function my_initialize_free_surface!(U, dev, Ny, u)
    copyto!(@view(U[1:112, 1:112, 1]), u[1:112, 1:112, 1])

    my_fill_halo_regions!(U, dev, Ny)

    return nothing
end

function my_fill_halo_regions!(c, dev, Ny)

    workgroup = KernelAbstractions.NDIteration.StaticSize{(16, 16)}()
    worksize  = KernelAbstractions.NDIteration.StaticSize{(112, 1)}()

    loop! = _my_fill_south_and_north_halo!(dev, workgroup, worksize)

    # Don't launch kernels with no size
    loop!(c, Ny)

    return nothing
end

@kernel function _my_fill_south_and_north_halo!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, Ny+1, k] = c[i, Ny, k]
end

@info "ru before initialization:"
@show maximum(abs.(convert(Array, parent(ru)) - parent(vu)))
@show maximum(abs.(convert(Array, parent(ru2)) - parent(vu)))
@show maximum(abs.(convert(Array, parent(ru3)) - parent(vu)))

Reactant.Serialization.export_to_enzymejax(
    my_initialize_free_surface!,
    rU, rdev, Ny, ru;
    output_dir=exported_jax_file_dir,
    compile_options=CompileOptions(raise=true),
)

@show "Reactant:"

@show @code_hlo my_initialize_free_surface!(rU, rdev, Ny, ru)

@jit my_initialize_free_surface!(rU, rdev, Ny, ru)
@jit my_initialize_free_surface!(rU2, rdev, Ny, ru2)
@jit my_initialize_free_surface!(rU3, rdev, Ny, ru3)
@show "Vanilla:"
my_initialize_free_surface!(vU, vdev, Ny, vu)

@info "What happens to ru after initialization:"
@show maximum(abs.(convert(Array, parent(ru)) - parent(vu)))
@show maximum(abs.(convert(Array, parent(ru2)) - parent(vu)))
@show maximum(abs.(convert(Array, parent(ru3)) - parent(vu)))


@info "After initialization (should be 0, or at most maybe machine precision, but there's a bug):"
rU  = convert(Array, parent(rU))
rU2 = convert(Array, parent(rU2))
rU3 = convert(Array, parent(rU3))
vU  = parent(vU)

@show maximum(abs.(rU - vU))
@show maximum(abs.(rU2 - vU))
@show maximum(abs.(rU3 - vU))
