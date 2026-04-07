using JLD2
using Oceananigans.Fields: interpolate!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Architectures: on_architecture

# Analytic atmospheric profiles for the prescribed atmosphere — including
# specific humidity, which `data_free_ocean_climate_model_init` zeros out.
# Names are prefixed with `_oc_` to avoid clashing with the top-level
# definitions in `data_free_ocean_climate_model.jl`.
_oc_zonal_wind(λ, φ) = 4 * sind(2φ)^2 - 2 * exp(-(abs(φ) - 12)^2 / 72)
_oc_sunlight(λ, φ)   = -200 - 600 * cosd(φ)^2
_oc_Tatm(λ, φ, z=0)  = 30 * cosd(φ)
# ~21 g/kg at the equator, ~3 g/kg at the poles (≈ 78% RH against _oc_Tatm)
_oc_qatm(λ, φ)       = 0.003 + 0.018 * cosd(φ)

function _set_atm_tracers!(T, q, u, ua, shortwave, Qs, Ta, qa)
    T .= Ta .+ 273.15
    q .= qa
    u .= ua
    shortwave .= Qs
    nothing
end

# Broadcasts host 2D arrays directly into the (possibly 4D, sharded)
# atmosphere parent arrays. The 2D inputs broadcast across the time
# dimension. Pass `parent(atmosphere.tracers.T)` etc. as the targets so
# that under Distributed{ReactantState} the sharding metadata is taken
# from the target, not from a separately-built Field.
function _set_atm_tracers_from_host!(atmT, atmq, atmu, atmQsw, Ta_arr, qa_arr, ua_arr, Qs_arr)
    atmT  .= Ta_arr .+ 273.15
    atmq  .= qa_arr
    atmu  .= ua_arr
    atmQsw .= Qs_arr
    nothing
end

const DEFAULT_BATHYMETRY_FILE =
    abspath(joinpath(@__DIR__, "..", "bathymetry_sixth_degree.jld2"))
const DEFAULT_INITIAL_CONDITIONS_FILE =
    abspath(joinpath(@__DIR__, "..", "ecco2_initial_conditions_sixth_degree.jld2"))

#####
##### Source-field loaders for the cached 1/6° artifacts.
##### These build CPU `Field`s on a 1/6° lat-lon source grid that lives only
##### on the host. The data is small (≤ 1 GB) and we use it as the source
##### of an `interpolate!` onto the target grid (which lives on `arch`).
#####

#=
The "ideal" pattern would be:
  1) Load JLD2 into a CPU `Field` on the native 1/6° grid
  2) `on_architecture(arch, cpu_field)` to move it to `arch`
  3) `interpolate!(model_field, native_data)`

That works on `CPU()` and on plain `ReactantState()`. Under
`Distributed{ReactantState}` step (3) currently fails: Reactant's MLIR
pass manager errors with "failed to run pass manager on module" while
lowering `gpu__interpolate!` for two sharded `LatitudeLongitudeGrid`s
(repro module saved to /tmp/sharded_interpolate_repro.mlir during
debugging on this machine).

Workaround used here: do `interpolate!` entirely on a CPU twin of the
target grid, then `set!(device_field, host_array)` to transfer the
already-target-shaped data onto the (possibly sharded) device field. The
host→device `set!` is a NoSharding→DimsSharding reshard, which Reactant
handles (resharding must remain enabled — see notes in the sharded run
script).
=#

"""
    _interpolated_bathymetry_array(bathymetry_file, Nx, Ny)

Read cached 1/6° bathymetry, build a CPU source `Field`, `interpolate!`
onto a CPU target-shaped `Field`, return the interpolated host array.
"""
function _interpolated_bathymetry_array(bathymetry_file::AbstractString,
                                        Nx::Int, Ny::Int)
    isfile(bathymetry_file) || throw(ArgumentError(
        "Bathymetry file $bathymetry_file not found.\n" *
        "Run: julia --project=. simulations/download_sixth_degree_artifacts.jl"))

    h_cached = jldopen(bathymetry_file) do file
        file["bottom_height"]
    end

    Nx_src, Ny_src = size(h_cached)
    cpu_src_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
                                         size = (Nx_src, Ny_src),
                                         halo = (8, 8),
                                         longitude = (0, 360),
                                         latitude  = (-80, 80),
                                         topology  = (Periodic, Bounded, Flat))

    cpu_src_field = Field{Center, Center, Nothing}(cpu_src_grid)
    set!(cpu_src_field, h_cached)
    fill_halo_regions!(cpu_src_field)

    cpu_dst_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
                                         size = (Nx, Ny),
                                         halo = (8, 8),
                                         longitude = (0, 360),
                                         latitude  = (-80, 80),
                                         topology  = (Periodic, Bounded, Flat))

    cpu_dst_field = Field{Center, Center, Nothing}(cpu_dst_grid)
    interpolate!(cpu_dst_field, cpu_src_field)
    fill_halo_regions!(cpu_dst_field)

    return Array(interior(cpu_dst_field, :, :, 1))
end

"""
    _interpolated_T_S_arrays(ic_file, Nx, Ny, Nz, target_z)

Read cached 1/6° T, S, build CPU source `Field`s, `interpolate!` onto
CPU target-shaped `Field`s, return interpolated host arrays.
"""
function _interpolated_T_S_arrays(ic_file::AbstractString,
                                  Nx::Int, Ny::Int, Nz::Int, target_z)
    isfile(ic_file) || throw(ArgumentError(
        "Initial-conditions file $ic_file not found.\n" *
        "Run: julia --project=. simulations/download_sixth_degree_artifacts.jl"))

    T_cached, S_cached, Nx_ic, Ny_ic, Nz_ic = jldopen(ic_file) do file
        (file["T"], file["S"], file["Nx"], file["Ny"], file["Nz"])
    end

    src_z = exponential_z_faces(; Nz = Nz_ic, depth = 4000, h = 30)
    cpu_src_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
                                         size = (Nx_ic, Ny_ic, Nz_ic),
                                         halo = (8, 8, 8),
                                         longitude = (0, 360),
                                         latitude  = (-80, 80),
                                         z = src_z)

    cpu_T_src = Field{Center, Center, Center}(cpu_src_grid); set!(cpu_T_src, T_cached); fill_halo_regions!(cpu_T_src)
    cpu_S_src = Field{Center, Center, Center}(cpu_src_grid); set!(cpu_S_src, S_cached); fill_halo_regions!(cpu_S_src)

    cpu_dst_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
                                         size = (Nx, Ny, Nz),
                                         halo = (8, 8, 8),
                                         longitude = (0, 360),
                                         latitude  = (-80, 80),
                                         z = target_z)

    cpu_T_dst = Field{Center, Center, Center}(cpu_dst_grid); interpolate!(cpu_T_dst, cpu_T_src); fill_halo_regions!(cpu_T_dst)
    cpu_S_dst = Field{Center, Center, Center}(cpu_dst_grid); interpolate!(cpu_S_dst, cpu_S_src); fill_halo_regions!(cpu_S_dst)

    return Array(interior(cpu_T_dst)), Array(interior(cpu_S_dst))
end

"""
    ocean_climate_model_init(arch=ReactantState();
                             resolution=2, Nz=20, Δt=30seconds,
                             bathymetry_file=DEFAULT_BATHYMETRY_FILE,
                             initial_conditions_file=DEFAULT_INITIAL_CONDITIONS_FILE)

Build a coupled ocean climate model with realistic bathymetry and initial
conditions, parallel in structure to `data_free_ocean_climate_model_init`.

Bathymetry and (T, S) ICs are loaded from cached 1/6° JLD2 artifacts and
**interpolated** onto the target grid via Oceananigans' `interpolate!`.
Download the artifacts beforehand with
`julia --project=. simulations/download_sixth_degree_artifacts.jl`.

The atmosphere is the same prescribed analytic atmosphere as
`data_free_ocean_climate_model_init`, **plus a prescribed specific
humidity profile** (which the data-free version zeros out).
"""
function ocean_climate_model_init(
    arch::Architectures.AbstractArchitecture = Architectures.ReactantState();
    resolution::Real = 2,
    Nz::Int = 20,
    Δt = 30seconds,
    bathymetry_file::AbstractString         = DEFAULT_BATHYMETRY_FILE,
    initial_conditions_file::AbstractString = DEFAULT_INITIAL_CONDITIONS_FILE,
)
    # ---- Underlying target grid ----
    Nx, Ny = resolution_to_points(resolution)
    z_target = exponential_z_faces(; Nz, depth = 4000, h = 30)
    underlying_grid = LatitudeLongitudeGrid(arch;
                                            size = (Nx, Ny, Nz),
                                            halo = (8, 8, 8),
                                            z = z_target,
                                            longitude = (0, 360),
                                            latitude  = (-80, 80))

    # ---- Bathymetry: CPU interpolate! onto target shape, then set! onto device ----
    bh_array = @gbprofile "interpolate_bathymetry_cpu" _interpolated_bathymetry_array(
        bathymetry_file, Nx, Ny)

    bottom_height = Field{Center, Center, Nothing}(underlying_grid)
    @gbprofile "set_bottom_height" set!(bottom_height, bh_array)
    fill_halo_regions!(bottom_height)

    grid = @gbprofile "ImmersedBoundaryGrid" ImmersedBoundaryGrid(
        underlying_grid,
        GridFittedBottom(bottom_height);
        active_cells_map = false,
    )

    # ---- Ocean simulation ----
    free_surface = SplitExplicitFreeSurface(substeps = 30)
    ocean = @gbprofile "ocean_simulation" ocean_simulation(grid; free_surface, Δt)

    # ---- Initial conditions: CPU interpolate! onto target shape, then set! onto device ----
    T_array, S_array = @gbprofile "interpolate_TS_cpu" _interpolated_T_S_arrays(
        initial_conditions_file, Nx, Ny, Nz, z_target)

    @gbprofile "set_T" set!(ocean.model.tracers.T, T_array)
    @gbprofile "set_S" set!(ocean.model.tracers.S, S_array)
    fill_halo_regions!(ocean.model.tracers.T)
    fill_halo_regions!(ocean.model.tracers.S)

    # ---- Prescribed analytic atmosphere (with humidity) ----
    atmos_times = range(0, 1days, length = 24)

    atmos_grid = LatitudeLongitudeGrid(arch,
                                       size = (360, 180),
                                       longitude = (0, 360),
                                       latitude  = (-90, 90),
                                       topology  = (Periodic, Bounded, Flat))

    atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times)

    # Compute the analytic profiles on a CPU twin grid (set!(field, fn)
    # does not currently compile under Distributed{ReactantState}). Then
    # broadcast the resulting host arrays directly into the
    # *atmosphere's own* parent arrays — under sharding the target's
    # sharding metadata governs the placement, instead of an
    # intermediate Ta/ua/Qs/qa field that would be replicated.
    cpu_atmos_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
                                           size = (360, 180),
                                           longitude = (0, 360),
                                           latitude  = (-90, 90),
                                           topology  = (Periodic, Bounded, Flat))
    cpu_Ta = Field{Center, Center, Nothing}(cpu_atmos_grid); set!(cpu_Ta, _oc_Tatm)
    cpu_ua = Field{Center, Center, Nothing}(cpu_atmos_grid); set!(cpu_ua, _oc_zonal_wind)
    cpu_Qs = Field{Center, Center, Nothing}(cpu_atmos_grid); set!(cpu_Qs, _oc_sunlight)
    cpu_qa = Field{Center, Center, Nothing}(cpu_atmos_grid); set!(cpu_qa, _oc_qatm)

    # Use parent (halo-included) so broadcast shape matches the
    # atmosphere's parent arrays on the target.
    Ta_arr = Array(parent(cpu_Ta))
    ua_arr = Array(parent(cpu_ua))
    Qs_arr = Array(parent(cpu_Qs))
    qa_arr = Array(parent(cpu_qa))

    # Distributed{ReactantState} also needs the @jit path — check via child arch.
    reactant_arch = arch isa Architectures.ReactantState ||
                    (arch isa Oceananigans.Distributed &&
                     Oceananigans.Architectures.child_architecture(arch) isa Architectures.ReactantState)

    if reactant_arch
        if Reactant.precompiling()
            @code_hlo _set_atm_tracers_from_host!(parent(atmosphere.tracers.T),
                                                  parent(atmosphere.tracers.q),
                                                  parent(atmosphere.velocities.u),
                                                  parent(atmosphere.downwelling_radiation.shortwave),
                                                  Ta_arr, qa_arr, ua_arr, Qs_arr)
        else
            @jit _set_atm_tracers_from_host!(parent(atmosphere.tracers.T),
                                             parent(atmosphere.tracers.q),
                                             parent(atmosphere.velocities.u),
                                             parent(atmosphere.downwelling_radiation.shortwave),
                                             Ta_arr, qa_arr, ua_arr, Qs_arr)
        end
    else
        _set_atm_tracers_from_host!(parent(atmosphere.tracers.T),
                                    parent(atmosphere.tracers.q),
                                    parent(atmosphere.velocities.u),
                                    parent(atmosphere.downwelling_radiation.shortwave),
                                    Ta_arr, qa_arr, ua_arr, Qs_arr)
    end

    # ---- Coupled model ----
    radiation = Radiation(arch)
    solver_stop_criteria = FixedIterations(5)
    atmosphere_ocean_flux_formulation = SimilarityTheoryFluxes(; solver_stop_criteria)
    interfaces = ComponentInterfaces(atmosphere, ocean; radiation, atmosphere_ocean_flux_formulation)
    coupled_model = @gbprofile "OceanSeaIceModel" OceanSeaIceModel(ocean; atmosphere, radiation, interfaces)

    return coupled_model
end
