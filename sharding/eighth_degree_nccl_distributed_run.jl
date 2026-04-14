# Vanilla-Oceananigans 1/8° atmosphere run — 8 GPUs with NCCLDistributed.
# Initialized from assembled 1/4° final state (interpolated 2× in horizontal).
# Latitude range (-70, 70), Δt=1s.
#
# Launch: ~/.julia/bin/mpiexecjl -n 8 --project julia -O0 sharding/eighth_degree_nccl_distributed_run.jl
#
# IMPORTANT: Does NOT load GordonBell25 or Reactant.

using Dates
using MPI
MPI.Init()

rank   = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

using JLD2
using Printf
using CUDA
using NCCL

# Assign one GPU per MPI rank
CUDA.device!(rank % length(CUDA.devices()))

rank == 0 && @info "Starting NCCL-distributed eighth-degree atmosphere simulation" nprocs now(UTC)
rank == 0 && @info "GPU assignment" rank gpu=CUDA.device()

using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.DistributedComputations: Distributed, Partition
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using KernelAbstractions: @kernel, @index
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics, ExplicitTimeStepping
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density, specific_prognostic_moisture
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics

Oceananigans.defaults.FloatType = Float32

# Load NCCLDistributed from the Oceananigans NCCL extension
const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributed = NCCLExt.NCCLDistributed

# ═══════════════════════════════════════════════════════════════════════════
# DCMIP-2016 balanced state (from src/moist_baroclinic_wave_model.jl)
# ═══════════════════════════════════════════════════════════════════════════

const earth_radius   = 6371220.0
const gravity        = 9.80616
const Rd_dry         = 287.0
const ε_virtual      = 0.608
const κ_exponent     = 2.0 / 7.0
const p_ref          = 1e5

const T_equator      = 310.0
const T_polar        = 240.0
const T_mean         = 0.5 * (T_equator + T_polar)
const lapse_rate     = 0.005
const jet_width      = 3.0
const vert_width     = 2.0

const coeff_A        = 1.0 / lapse_rate
const coeff_B        = (T_mean - T_polar) / (T_mean * T_polar)
const coeff_C        = 0.5 * (jet_width + 2) * (T_equator - T_polar) / (T_equator * T_polar)
const scale_height   = Rd_dry * T_mean / gravity

const q0_surface    = 0.018
const φ_width       = 2π / 9
const p_width       = 34000.0
const η_tropopause  = 0.1
const q_tropopause  = 1e-12

function vertical_structure(z)
    ζ      = z / (vert_width * scale_height)
    exp_ζ² = exp(-ζ^2)
    τ₁ = coeff_A * lapse_rate / T_mean * exp(lapse_rate * z / T_mean) +
         coeff_B * (1 - 2 * ζ^2) * exp_ζ²
    τ₂ = coeff_C * (1 - 2 * ζ^2) * exp_ζ²
    I₁ = coeff_A * (exp(lapse_rate * z / T_mean) - 1) +
         coeff_B * z * exp_ζ²
    I₂ = coeff_C * z * exp_ζ²
    return (; τ₁, τ₂, I₁, I₂)
end

F_temperature(cosφ) = cosφ^jet_width - jet_width / (jet_width + 2) * cosφ^(jet_width + 2)

function virtual_temperature(φ, z)
    vs = vertical_structure(z)
    return 1.0 / (vs.τ₁ - vs.τ₂ * F_temperature(cos(φ)))
end

function balanced_pressure(φ, z)
    vs = vertical_structure(z)
    return p_ref * exp(-gravity / Rd_dry * (vs.I₁ - vs.I₂ * F_temperature(cos(φ))))
end

function moisture_profile(φ, z)
    p = balanced_pressure(φ, z)
    η = p / p_ref
    q_below = q0_surface * exp(-(φ / φ_width)^4) *
                           exp(-((η - 1) * p_ref / p_width)^2)
    return ifelse(η > η_tropopause, q_below, q_tropopause)
end

function initial_theta(λ_deg, φ_deg, z)
    φ  = deg2rad(φ_deg)
    Tv = virtual_temperature(φ, z)
    p  = balanced_pressure(φ, z)
    q  = moisture_profile(φ, z)
    T  = Tv / (1 + ε_virtual * q)
    return T * (p_ref / p)^κ_exponent
end

theta_reference(z)               = initial_theta(0.0, 0.0, z)
surface_temperature(λ_deg, φ_deg) = virtual_temperature(deg2rad(φ_deg), 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# IC loader — each rank loads the full file, sets its local portion via
# interpolation from the source (quarter-degree) grid.
# ═══════════════════════════════════════════════════════════════════════════

function load_ic_distributed!(model, path::String;
                              H = 30e3,
                              src_latitude  = (-70, 70),
                              src_longitude = (0, 360))
    Nλ_src, Nφ_src, Nz_src, ρ_data, ρu_data, ρv_data, ρw_data, ρθ_data, ρqv_data =
        JLD2.jldopen(path, "r") do file
            (file["Nλ"], file["Nφ"], file["Nz"],
             file["ρ"], file["ρu"], file["ρv"], file["ρw"],
             file["ρθ"], file["ρqᵛ"])
        end

    ρqcl_data, ρqci_data = JLD2.jldopen(path, "r") do file
        ρqcl = haskey(file, "micro_ρqᶜˡ") ? file["micro_ρqᶜˡ"] : nothing
        ρqci = haskey(file, "micro_ρqᶜⁱ") ? file["micro_ρqᶜⁱ"] : nothing
        (ρqcl, ρqci)
    end

    grid = model.grid
    arch = Oceananigans.architecture(grid)
    FT   = eltype(grid)

    pairs = Any[
        (ρ_data,   dynamics_density(model.dynamics)),
        (ρu_data,  model.momentum.ρu),
        (ρv_data,  model.momentum.ρv),
        (ρw_data,  model.momentum.ρw),
        (ρθ_data,  model.formulation.potential_temperature_density),
        (ρqv_data, model.moisture_density),
    ]

    if ρqcl_data !== nothing
        push!(pairs, (ρqcl_data, model.microphysical_fields[:ρqᶜˡ]))
    end
    if ρqci_data !== nothing
        push!(pairs, (ρqci_data, model.microphysical_fields[:ρqᶜⁱ]))
    end

    halo = Oceananigans.halo_size(grid)

    # Build a global source grid on GPU for interpolation
    src_grid = LatitudeLongitudeGrid(GPU();
        size = (Nλ_src, Nφ_src, Nz_src),
        halo = halo,
        latitude  = src_latitude,
        longitude = src_longitude,
        z = (0, H))

    for (src_array, target_field) in pairs
        loc = Oceananigans.location(target_field)
        iloc = map(L -> L(), loc)
        src_field = Field(iloc, src_grid)
        gpu_data = Oceananigans.on_architecture(GPU(), Array{FT}(src_array))
        copyto!(Oceananigans.interior(src_field), gpu_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(src_field)

        rank == 0 && @info "interpolate!" field=nameof(typeof(target_field)) loc src=size(Oceananigans.interior(src_field)) dst=size(Oceananigans.interior(target_field))
        Oceananigans.Fields.interpolate!(target_field, src_field)
    end

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Model construction
# ═══════════════════════════════════════════════════════════════════════════

function build_atmosphere_model(arch; Nλ, Nφ, Nz, H, Δt, halo, latitude)
    grid = LatitudeLongitudeGrid(arch;
                                 size = (Nλ, Nφ, Nz),
                                 halo,
                                 longitude = (0, 360),
                                 latitude,
                                 z = (0, H))

    coriolis = SphericalCoriolis()

    dynamics = CompressibleDynamics(
        ExplicitTimeStepping();
        surface_pressure = p_ref,
        reference_potential_temperature = theta_reference,
    )

    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    FT = Oceananigans.defaults.FloatType
    rate = FT(1) / FT(30)
    cf = Breeze.Microphysics.ConstantRateCondensateFormation{FT}(rate)
    cloud_formation = NonEquilibriumCloudFormation(cf, cf)
    microphysics = ext.OneMomentCloudMicrophysics(; cloud_formation)

    weno     = WENO(order = 5)
    weno_pos = WENO(order = 5, bounds = (0, 1))
    momentum_advection = weno
    scalar_advection = (ρθ   = weno,
                        ρqᵛ  = weno_pos,
                        ρqᶜˡ = weno_pos,
                        ρqᶜⁱ = weno_pos,
                        ρqʳ  = weno_pos,
                        ρqˢ  = weno_pos)

    Cᴰ = 1e-3
    Uᵍ = 1e-2
    T₀ = surface_temperature

    ρu_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρv_bcs  = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρe_bcs  = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))
    ρqᵛ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀))

    boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵛ=ρqᵛ_bcs)

    model = AtmosphereModel(grid; dynamics, coriolis, momentum_advection,
                            scalar_advection, microphysics, boundary_conditions)

    FT_grid = eltype(grid)
    model.clock.last_Δt = FT_grid(Δt)

    return model
end

# GPU-side NaN check
function any_nan(model)
    fields = (dynamics_density(model.dynamics),
              model.momentum.ρu, model.momentum.ρv, model.momentum.ρw,
              model.formulation.potential_temperature_density,
              model.moisture_density)
    for f in fields
        any(isnan, parent(f)) && return true
    end
    return false
end

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

Rx = 4
Ry = 2
@assert nprocs == Rx * Ry "Expected $(Rx * Ry) MPI ranks, got $nprocs"

rank == 0 && @info "Setting up NCCLDistributed" Rx Ry
arch = NCCLDistributed(GPU(); partition = Partition(Rx, Ry, 1))

Nλ = 2880
Nφ = 1120
Nz = 64
column_height = 30e3
Δt = 1.0

ic_path = joinpath(@__DIR__, "..", "simulations", "initial_conditions",
                   "quarter_deg_3h_final.jld2")
isfile(ic_path) || error("IC file not found at $ic_path")
rank == 0 && @info "IC file (1/4° → 1/8° interpolation)" ic_path

rank == 0 && @info "Building model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz, lat=(-70,70), Δt=$(Δt)s)..." now(UTC)
@time "build model" model = build_atmosphere_model(arch; Nλ, Nφ, Nz,
                                                    H=column_height, Δt,
                                                    halo=(4, 4, 4),
                                                    latitude=(-70, 70))
rank == 0 && @show model

rank == 0 && @info "Loading ICs (interpolating from 1/4° assembled state)..." now(UTC)
@time "load ICs" load_ic_distributed!(model, ic_path;
                                       H=column_height,
                                       src_latitude=(-70, 70),
                                       src_longitude=(0, 360))

if any_nan(model)
    error("NaN after IC load on rank $rank")
end
rank == 0 && @info "Post-load: no NaN"

# ── First time step ────────────────────────────────────────────────────

rank == 0 && @info "First time step..." now(UTC)
@time "first step" begin
    Oceananigans.TimeSteppers.update_state!(model)
    Oceananigans.TimeSteppers.time_step!(model, Δt)
end

if any_nan(model)
    error("NaN after first time step on rank $rank")
end
rank == 0 && @info "First step complete" model.clock

# ── Output + Simulation setup ────────────────────────────────────────

output_dir = joinpath(@__DIR__, "..", "simulations", "output", "nccl_8gpu_eighth_deg")
rank == 0 && mkpath(output_dir)
MPI.Barrier(MPI.COMM_WORLD)

output_fields = Oceananigans.fields(model)
output_prefix = joinpath(output_dir, "fields")

stop_iter = 10800  # 3h with Δt=1s

simulation = Simulation(model; Δt, stop_iteration=stop_iter)

simulation.output_writers[:fields] = JLD2Writer(model, output_fields;
    filename = output_prefix,
    schedule = IterationInterval(2048),
    overwrite_existing = true)

# Progress callback with NaN check
wall_start = Ref(time_ns())
function progress(sim)
    nan_now = any_nan(sim.model)
    if rank == 0
        wall = (time_ns() - wall_start[]) / 1e9
        sim_days = sim.model.clock.time / 86400
        sdpd = sim_days / (wall / 86400)
        @info @sprintf("iter %d, sim_time=%.0fs (day %.3f), SDPD=%.1f, wall=%.0fs, NaN=%s",
                        sim.model.clock.iteration, sim.model.clock.time,
                        sim_days, sdpd, wall, nan_now)
    end
    nan_now && error("NaN detected at iteration $(sim.model.clock.iteration) on rank $rank")
    flush(stderr); flush(stdout)
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(512))

rank == 0 && @info "Starting simulation (stop_iter=$stop_iter, Δt=$Δt)" now(UTC)
wall_start[] = time_ns()
Oceananigans.run!(simulation)

rank == 0 && @info "Done!" now(UTC) model.clock
MPI.Finalize()
