using Oceananigans.Grids: halo_size
using Oceananigans.Utils: launch!, KernelParameters
using Oceananigans.Architectures: architecture
using KernelAbstractions: @kernel, @index

struct NeareastNeighbor end
struct LinearInterpolation end

@inline function initial_buoyancy(λ, φ, z)
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    Δφ = 20
    γ = π/2 - 2π * (φ₀ - φ) / Δφ
    μ = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return N² * z + Δb * μ + 1e-2 * Δb * randn()
end

function baroclinic_instability_model(arch; resolution, Nz, kw...)
    Nx, Ny = resolution_to_points(resolution)
    return baroclinic_instability_model(arch, Nx, Ny, Nz; kw...)
end

function baroclinic_instability_model(arch, Nx, Ny, Nz; Δt,
    initial_conditions_path::Union{Nothing,String} = joinpath(dirname(@__DIR__), "simulations", "initial_conditions", "baroclinic_100day_quarter_degree.jld2"),
    halo = (8, 8, 8),
    interpolation_mode = NeareastNeighbor(),
    grid_type = :simple_lat_lon, # :gaussian_islands

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30),

    # TEOS10 is a 54-term polynomial that relates temperature (T),
    # and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(
        equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType)),

    closure = nothing,    
    # closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity(),
    # closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1e-5, ν=1e-4),

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis(),

    # Simple momentum advection schemes. May need to be reconsidered
    # due to Float32.
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    )

    tracers = if buoyancy isa BuoyancyTracer
        [:b]
    elseif buoyancy isa SeawaterBuoyancy
        [:T, :S]
    else
        Symbol[]
    end

    if closure isa Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity
        push!(tracers, :e)
    elseif closure isa Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity
        push!(tracers, :e)
        push!(tracers, :ϵ)
    end

    tracers = tuple(tracers...)

    grid = if grid_type === :gaussian_islands
        gaussian_islands_tripolar_grid(arch, Nx, Ny, Nz; halo)
    elseif grid_type === :simple_lat_lon
        simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo)
    else
        error("grid_type=$grid_type must be :gaussian_islands or :simple_lat_lon.")
    end

    model = HydrostaticFreeSurfaceModel(;
        grid, free_surface, closure, buoyancy, tracers,
        coriolis, momentum_advection, tracer_advection,
    )

    Random.seed!(42)

    #=
    if buoyancy isa SeawaterBuoyancy
        set_baroclinic_instability!(model)
    elseif buoyancy isa BuoyancyTracer
        # set!(model, b=initial_buoyancy)
    end
    =#

    model.clock.last_Δt = Δt

    if initial_conditions_path !== nothing
        set_baroclinic_instability_from_file!(model, initial_conditions_path, interpolation_mode)
    end

    return model
end

@kernel function _nearest_neighbor_data_copy!(Sp, Tp, Rx, Ry, Hz, S_data, T_data)
    i, j, k = @index(Global, NTuple)
    i′ = ceil(Int, i / Rx)
    j′ = ceil(Int, j / Ry)
    @inbounds begin
        Sp[i, j, k+Hz] = S_data[i′, j′, k]
        Tp[i, j, k+Hz] = T_data[i′, j′, k]
    end
end

function set_baroclinic_instability_from_file!(model, path::String, mode::NeareastNeighbor)
    grid = model.grid
    
    model_size = size(grid)
    halos = halo_size(grid)

    Nx_src, Ny_src, Nz_src, T_data, S_data = JLD2.jldopen(path, "r") do file
        (file["Nx"], file["Ny"], file["Nz"], file["T"], file["S"])
    end

    if Nz_src != model_size[3]
        throw("The grid levels and the data levels need to be the same!")
    end

    @info "src size: $((Nx_src, Ny_src, Nz_src))"

    Rx = (model_size[1] + 2halos[1]) / Nx_src
    Ry = (model_size[2] + 2halos[2]) / Ny_src

    @info "model $((Rx, Ry)) times larger than initial condition in x, y, respectively"

    Px = 1:model_size[1]+2halos[1]
    Py = 1:model_size[2]+2halos[2]
    Pz = 1:model_size[3]

    Sp = parent(model.tracers.S)
    Tp = parent(model.tracers.T)

    @jit raise=true launch!(architecture(grid), grid, KernelParameters(Px, Py, Pz), _nearest_neighbor_data_copy!, Sp, Tp, Rx, Ry, halos[3], S_data, T_data)

    return nothing
end

function set_baroclinic_instability_from_file!(model, path::String, mode::LinearInterpolation)
    Nx_src, Ny_src, Nz_src, T_data, S_data = JLD2.jldopen(path, "r") do file
        (file["Nx"], file["Ny"], file["Nz"], file["T"], file["S"])
    end

    expected = (Nx_src, Ny_src, Nz_src)
    if size(T_data) != expected
        error("Loaded T field size $(size(T_data)) does not match (Nx, Ny, Nz) = $expected from $path")
    end
    if size(S_data) != expected
        error("Loaded S field size $(size(S_data)) does not match (Nx, Ny, Nz) = $expected from $path")
    end

    # Always build the source field on plain CPU. The pirated `interpolate!`
    # method below (target::ShardedField, source on CPU) takes care of feeding
    # this CPU source into a Reactant-sharded target via a one-shot @compile +
    # free_exec dance. For non-sharded targets (plain CPU / CUDA / single-device
    # Reactant) Oceananigans' stock `interpolate!` handles the CPU→arch case.
    #
    # Uniform z so that the source grid is `ZRegularLLG` and
    # `fractional_z_index` is constant-time (no binary search). The
    # binary-search path uses a dynamic-trip-count `while` loop that
    # Reactant's MLIR pass manager currently fails to lower under sharding.
    source_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
        size = (Nx_src, Ny_src, Nz_src),
        halo = (1, 1, 1),
        z = (-4000, 0),
        latitude = (-80, 80),
        longitude = (0, 360),
    )

    T_src = CenterField(source_grid)
    S_src = CenterField(source_grid)
    set!(T_src, T_data)
    set!(S_src, S_data)
    Oceananigans.BoundaryConditions.fill_halo_regions!(T_src)
    Oceananigans.BoundaryConditions.fill_halo_regions!(S_src)

    # Reactant `resize_linear!` is fast when the source dims are an integer
    # factor of the parent (target) grid in every dimension. If not, do a CPU-
    # side interpolation first onto an intermediate grid of size
    # (Nx/Rx, Ny/Ry, Nz) — which is by construction an integer factor of the
    # global target — and feed that to the Reactant resize.
    target_arch = Oceananigans.Architectures.architecture(model.grid)
    Rx, Ry = if target_arch isa Oceananigans.DistributedComputations.Distributed
        (target_arch.partition.x, target_arch.partition.y)
    else
        (1, 1)
    end
    # Compare TOTAL field sizes (interior + halos), not interior — Reactant's
    # `resize_linear!` operates on the full parent arrays.
    src_total = size(Oceananigans.interior(T_src))
    dst_total = size(Oceananigans.interior(model.tracers.T))
    is_int_factor(s, t) = (t % s == 0) || (s % t == 0)
    needs_cpu_interp = !all(is_int_factor.(src_total, dst_total))

    Nx_dst, Ny_dst, Nz_dst = size(model.grid)
    T_for_jit, S_for_jit = if needs_cpu_interp
        Nx_int = Nx_dst ÷ Rx
        Ny_int = Ny_dst ÷ Ry
        Nz_int = Nz_dst
        intermediate_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
            size = (Nx_int, Ny_int, Nz_int),
            halo = (1, 1, 1),
            z = (-4000, 0),
            latitude = (-80, 80),
            longitude = (0, 360),
        )
        T_int = CenterField(intermediate_grid)
        S_int = CenterField(intermediate_grid)
        Oceananigans.Fields.interpolate!(T_int, T_src)
        Oceananigans.Fields.interpolate!(S_int, S_src)
        Oceananigans.BoundaryConditions.fill_halo_regions!(T_int)
        Oceananigans.BoundaryConditions.fill_halo_regions!(S_int)
        (T_int, S_int)
    else
        (T_src, S_src)
    end

    src_sharding = if target_arch isa Oceananigans.DistributedComputations.Distributed
        mesh = Sharding.unwrap_shardinfo(Reactant.ancestor(model.tracers.T).sharding).mesh
        Sharding.Replicated(mesh)
    else
        Sharding.NoSharding()
    end
    to_ra(a::AbstractArray) = Reactant.to_rarray(a; sharding=src_sharding)

    function interpolate_both_fields!(
        T, S, T_src, S_src
    )
        interpolate_3d!(T, T_src)
        interpolate_3d!(S, S_src)
        return nothing
    end

    T_src = to_ra(Array(interior(T_for_jit)))
    S_src = to_ra(Array(interior(S_for_jit)))
    @jit interpolate_both_fields!(model.tracers.T, model.tracers.S, T_src, S_src)

    # println(@code_hlo interpolate_3d!(model.tracers.T, Reactant.to_rarray(ones(
    #                                                             eltype(model.tracers.T),
    #                                                             size(model.tracers.T) .÷ 2
    #                                                                           ))))
    # println(@code_hlo interpolate_3d!(model.tracers.S, Reactant.to_rarray(S_src)))

    return nothing
end

# ----------------------------------------------------------------------------
# Sharded-Reactant interpolate! pirate
#
# Type aliases mirroring OceananigansReactantExt.{Grids,Fields} but defined
# against core Oceananigans types so they're available without the extension
# being loaded (the extension only loads its own copies; both refer to the
# same underlying core types so dispatch is identical).
const ShardedDistributedArch   = Oceananigans.DistributedComputations.Distributed{<:Oceananigans.Architectures.ReactantState}
const ShardedGrid{FT,TX,TY,TZ} = Oceananigans.Grids.AbstractGrid{FT,TX,TY,TZ,<:ShardedDistributedArch}
const ShardedField{LX,LY,LZ,O} = Oceananigans.Fields.Field{LX,LY,LZ,O,<:ShardedGrid}

const CPUSourceGrid            = Oceananigans.Grids.AbstractGrid{<:Any,<:Any,<:Any,<:Any,<:Oceananigans.Architectures.CPU}
const CPUSourceField{LX,LY,LZ,O} = Oceananigans.Fields.Field{LX,LY,LZ,O,<:CPUSourceGrid}

# Mirrors the kernel-launch body of `Oceananigans.Fields.interpolate!` (see
# Oceananigans.jl/src/Fields/interpolate.jl), without the
# `to_arch != from_arch` assertion. Called both directly (from this method)
# and as the body that Reactant traces under @compile.
function _gb25_interpolate_kernel!(to_field, from_field)
    to_grid   = to_field.grid
    from_grid = from_field.grid
    to_arch   = Oceananigans.Architectures.child_architecture(Oceananigans.Architectures.architecture(to_field))

    from_location = Tuple(L() for L in Oceananigans.Fields.location(from_field))
    to_location   = Tuple(L() for L in Oceananigans.Fields.location(to_field))

    params = Oceananigans.Utils.KernelParameters(Oceananigans.Fields.interior_indices(to_field))

    Oceananigans.Utils.launch!(to_arch, to_grid, params,
        Oceananigans.Fields._interpolate!, to_field, to_grid, to_location,
        from_field, from_grid, from_location)

    Oceananigans.BoundaryConditions.fill_halo_regions!(to_field)
    return to_field
end

# Type piracy on Oceananigans.Fields.interpolate! for the case
#   target = sharded Reactant field (Distributed{ReactantState})
#   source = plain CPU field
# The stock method errors with `to_arch != from_arch`. We bypass that and run
# the interpolation kernel under @compile so it lowers to a single XLA program;
# the executable is freed immediately so the one-shot compile doesn't leak for
# the rest of the session.
function Oceananigans.Fields.interpolate!(target::ShardedField, source::CPUSourceField)
    compiled = @compile sync = true raise = true _gb25_interpolate_kernel!(target, source)
    compiled(target, source)
    Reactant.XLA.IFRT.free_exec(compiled.exec)
    compiled.exec.exec = C_NULL
    return target
end
