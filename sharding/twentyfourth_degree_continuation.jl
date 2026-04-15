# 1/24° continuation run — restarts from the iter27000 (t=6h) checkpoint
# of the prior 1/24° run and integrates another 6h with 1h save cadence.
# Iteration-based save check (no float drift) so the final state is guaranteed.
#
# Launch: ~/.julia/bin/mpiexecjl -n 8 --project julia -O0 sharding/twentyfourth_degree_continuation.jl

using Dates, MPI, JLD2, Printf, CUDA, NCCL
MPI.Init()
rank   = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
CUDA.device!(rank % length(CUDA.devices()))

rank == 0 && @info "Starting 1/24° NCCL simulation" nprocs now(UTC)

using Oceananigans
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField
using Oceananigans.DistributedComputations: Distributed, Partition
using KernelAbstractions: @kernel, @index
using Breeze
using Breeze: AtmosphereModel, CompressibleDynamics, ExplicitTimeStepping
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
using Breeze.AtmosphereModels: dynamics_density
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics

Oceananigans.defaults.FloatType = Float32

const NCCLExt = Base.get_extension(Oceananigans, :OceananigansNCCLExt)
const NCCLDistributed = NCCLExt.NCCLDistributed

# ═══════════════════════════════════════════════════════════════════════════
# DCMIP-2016 reference state (dynamics + surface BCs only; IC from file)
# ═══════════════════════════════════════════════════════════════════════════

const Rd_dry = 287.0; const gravity = 9.80616; const p_ref = 1e5
const T_equator = 310.0; const T_polar = 240.0; const T_mean = 0.5 * (T_equator + T_polar)
const lapse_rate = 0.005; const jet_width = 3.0; const vert_width = 2.0
const ε_virtual = 0.608; const κ_exponent = 2.0/7.0
const coeff_A = 1.0/lapse_rate
const coeff_B = (T_mean - T_polar)/(T_mean * T_polar)
const coeff_C = 0.5 * (jet_width + 2) * (T_equator - T_polar) / (T_equator * T_polar)
const scale_height = Rd_dry * T_mean / gravity
const q0_surface = 0.018; const φ_width = 2π/9; const p_width = 34000.0
const η_tropopause = 0.1; const q_tropopause = 1e-12

function vertical_structure(z)
    ζ = z / (vert_width * scale_height); e = exp(-ζ^2)
    τ₁ = coeff_A*lapse_rate/T_mean*exp(lapse_rate*z/T_mean) + coeff_B*(1-2ζ^2)*e
    τ₂ = coeff_C*(1-2ζ^2)*e
    I₁ = coeff_A*(exp(lapse_rate*z/T_mean)-1) + coeff_B*z*e
    I₂ = coeff_C*z*e
    return (; τ₁, τ₂, I₁, I₂)
end
F_temperature(cosφ) = cosφ^jet_width - jet_width/(jet_width+2)*cosφ^(jet_width+2)
function virtual_temperature(φ, z)
    vs = vertical_structure(z); 1.0/(vs.τ₁ - vs.τ₂*F_temperature(cos(φ)))
end
function balanced_pressure(φ, z)
    vs = vertical_structure(z); p_ref*exp(-gravity/Rd_dry*(vs.I₁ - vs.I₂*F_temperature(cos(φ))))
end
function moisture_profile(φ, z)
    p = balanced_pressure(φ, z); η = p/p_ref
    q = q0_surface*exp(-(φ/φ_width)^4)*exp(-((η-1)*p_ref/p_width)^2)
    ifelse(η > η_tropopause, q, q_tropopause)
end
function initial_theta(λ_deg, φ_deg, z)
    φ = deg2rad(φ_deg); Tv = virtual_temperature(φ, z)
    p = balanced_pressure(φ, z); q = moisture_profile(φ, z)
    T = Tv/(1 + ε_virtual*q); T*(p_ref/p)^κ_exponent
end
theta_reference(z) = initial_theta(0.0, 0.0, z)
surface_temperature(λ, φ) = virtual_temperature(deg2rad(φ), 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# IC loader
# ═══════════════════════════════════════════════════════════════════════════

function load_ic_same_resolution!(model, path::String; Rx::Int, Ry::Int)
    # Direct tile copy from assembled file — no interpolation. Source grid
    # resolution must match the distributed target grid exactly. Reads only
    # this rank's slice from the file (no full-global GPU allocation).
    ix = rank ÷ Ry; iy = rank % Ry
    Nλ, Nφ, Nz = JLD2.jldopen(path, "r") do f; Int(f["Nλ"]), Int(f["Nφ"]), Int(f["Nz"]); end
    Nx_per = Nλ ÷ Rx; Ny_per = Nφ ÷ Ry
    x_range = (ix * Nx_per + 1):((ix + 1) * Nx_per)
    y_range_c = (iy * Ny_per + 1):((iy + 1) * Ny_per)
    # ρv: top-y rank includes the wall face (Ny_per + 1 rows)
    y_range_v = (iy == Ry - 1) ? ((iy * Ny_per + 1):((iy + 1) * Ny_per + 1)) : y_range_c

    function load_one!(target, key, yr; z_range=Colon())
        tile = JLD2.jldopen(path, "r") do f
            Array{Float32}(f[key][x_range, yr, z_range])
        end
        gpu_tile = Oceananigans.on_architecture(GPU(), tile)
        copyto!(Oceananigans.interior(target), gpu_tile)
        Oceananigans.BoundaryConditions.fill_halo_regions!(target)
        rank == 0 && @info "loaded" key=key tile=size(tile)
    end

    load_one!(dynamics_density(model.dynamics), "ρ", y_range_c)
    load_one!(model.momentum.ρu, "ρu", y_range_c)
    load_one!(model.momentum.ρv, "ρv", y_range_v)
    load_one!(model.momentum.ρw, "ρw", y_range_c)  # ρw has Nz+1 in z, all levels
    load_one!(model.formulation.potential_temperature_density, "ρθ", y_range_c)
    load_one!(model.moisture_density, "ρqᵛ", y_range_c)
    for (file_key, field_name) in [("micro_ρqᶜˡ", :ρqᶜˡ), ("micro_ρqᶜⁱ", :ρqᶜⁱ),
                                    ("ρqʳ", :ρqʳ), ("ρqˢ", :ρqˢ)]
        haskey(model.microphysical_fields, field_name) || continue
        load_one!(model.microphysical_fields[field_name], file_key, y_range_c)
    end
    rank == 0 && @info "IC loaded (direct-copy, no interp)"
    return nothing
end

function load_ic!(model, path::String; H=30e3, src_latitude=(-80,80), src_longitude=(0,360))
    Nλ_src, Nφ_src, Nz_src, ρ_d, ρu_d, ρv_d, ρw_d, ρθ_d, ρqv_d =
        JLD2.jldopen(path, "r") do f
            (f["Nλ"], f["Nφ"], f["Nz"],
             f["ρ"], f["ρu"], f["ρv"], f["ρw"], f["ρθ"], f["ρqᵛ"])
        end

    ρqcl_d, ρqci_d, ρqr_d, ρqs_d = JLD2.jldopen(path, "r") do f
        (haskey(f,"micro_ρqᶜˡ") ? f["micro_ρqᶜˡ"] : nothing,
         haskey(f,"micro_ρqᶜⁱ") ? f["micro_ρqᶜⁱ"] : nothing,
         haskey(f,"ρqʳ") ? f["ρqʳ"] : nothing,
         haskey(f,"ρqˢ") ? f["ρqˢ"] : nothing)
    end

    grid = model.grid; FT = eltype(grid)
    pairs = Any[
        (ρ_d,  dynamics_density(model.dynamics)),
        (ρu_d, model.momentum.ρu), (ρv_d, model.momentum.ρv),
        (ρw_d, model.momentum.ρw),
        (ρθ_d, model.formulation.potential_temperature_density),
        (ρqv_d, model.moisture_density),
    ]
    for (d, name) in [(ρqcl_d,:ρqᶜˡ),(ρqci_d,:ρqᶜⁱ),(ρqr_d,:ρqʳ),(ρqs_d,:ρqˢ)]
        d !== nothing && push!(pairs, (d, model.microphysical_fields[name]))
    end

    halo = Oceananigans.halo_size(grid)
    src_grid = LatitudeLongitudeGrid(GPU();
        size=(Nλ_src, Nφ_src, Nz_src), halo=halo,
        latitude=src_latitude, longitude=src_longitude, z=(0, H))

    for (src_array, target_field) in pairs
        loc = Oceananigans.location(target_field)
        iloc = map(L -> L(), loc)
        src_field = Field(iloc, src_grid)
        gpu_data = Oceananigans.on_architecture(GPU(), Array{FT}(src_array))
        copyto!(Oceananigans.interior(src_field), gpu_data)
        Oceananigans.BoundaryConditions.fill_halo_regions!(src_field)
        rank == 0 && @info "interpolate!" field=nameof(typeof(target_field)) src=size(Oceananigans.interior(src_field)) dst=size(Oceananigans.interior(target_field))
        Oceananigans.Fields.interpolate!(target_field, src_field)
        # Fill TARGET halos after interpolation — without this, halos contain
        # zeros/garbage which WENO reads and produces NaN at boundary cells.
        Oceananigans.BoundaryConditions.fill_halo_regions!(target_field)
    end

    # NO moisture clamping — preserves the balanced state
    rank == 0 && @info "No moisture clamping (preserving balanced state)"
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════════

function build_model(arch; Nλ, Nφ, Nz, H, Δt, halo, latitude, sst_anomaly=0, τ_cloud=120)
    grid = LatitudeLongitudeGrid(arch; size=(Nλ,Nφ,Nz), halo, longitude=(0,360), latitude, z=(0,H))
    dynamics = CompressibleDynamics(ExplicitTimeStepping(); surface_pressure=p_ref,
                                    reference_potential_temperature=theta_reference)
    FT = Oceananigans.defaults.FloatType
    rate = FT(1)/FT(τ_cloud)
    cf = Breeze.Microphysics.ConstantRateCondensateFormation{FT}(rate)
    microphysics = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt).OneMomentCloudMicrophysics(;
        cloud_formation=NonEquilibriumCloudFormation(cf, cf))
    weno = WENO(order=5); weno_pos = WENO(order=5, bounds=(0,1))
    scalar_advection = (ρθ=weno, ρqᵛ=weno_pos, ρqᶜˡ=weno_pos, ρqᶜⁱ=weno_pos, ρqʳ=weno_pos, ρqˢ=weno_pos)
    Cᴰ=1e-3; Uᵍ=1e-2; T₀=(λ,φ)->surface_temperature(λ,φ)+sst_anomaly
    boundary_conditions = (;
        ρu=FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀)),
        ρv=FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀)),
        ρe=FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀)),
        ρqᵛ=FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=Cᴰ, gustiness=Uᵍ, surface_temperature=T₀)))
    model = AtmosphereModel(grid; dynamics, coriolis=SphericalCoriolis(),
                            momentum_advection=weno, scalar_advection, microphysics, boundary_conditions)
    model.clock.last_Δt = FT(Δt)
    return model
end

function any_nan(model)
    for f in (dynamics_density(model.dynamics), model.momentum.ρu, model.momentum.ρv,
              model.momentum.ρw, model.formulation.potential_temperature_density, model.moisture_density)
        any(isnan, parent(f)) && return true
    end
    return false
end

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

function field_extrema(f)
    p = Oceananigans.interior(f)
    Float64(minimum(p)), Float64(maximum(p))
end

wall_start = Ref(time_ns())

function diagnostics(sim)
    m = sim.model
    ρ_min, ρ_max = field_extrema(dynamics_density(m.dynamics))
    ρw_min, ρw_max = field_extrema(m.momentum.ρw)
    nan_here = !isfinite(ρ_min) || !isfinite(ρ_max) || !isfinite(ρw_min) || !isfinite(ρw_max)
    wall = (time_ns() - wall_start[]) / 1e9
    @info @sprintf("[r%d] iter %5d  t=%8.1fs  Δt=%.2f  wall=%6.1fs  ρ=[%.3e,%.3e] ρw=[%.2e,%.2e]",
                   rank, m.clock.iteration, m.clock.time, sim.Δt, wall, ρ_min, ρ_max, ρw_min, ρw_max)
    nan_here && error("NaN at iter $(m.clock.iteration) on rank $rank")
    flush(stderr); flush(stdout)
end

function save_output(sim, output_prefix, save_iter_interval)
    m = sim.model
    iter = m.clock.iteration
    iter > 0 && mod(iter, save_iter_interval) == 0 || return
    t = m.clock.time
    filepath = output_prefix * "_iter$(lpad(iter, 6, '0')).jld2"
    JLD2.jldopen(filepath, "w") do file
        file["iteration"] = iter; file["time"] = t; file["Δt"] = sim.Δt
        file["ρ"]  = Array(Oceananigans.interior(dynamics_density(m.dynamics)))
        file["ρu"] = Array(Oceananigans.interior(m.momentum.ρu))
        file["ρv"] = Array(Oceananigans.interior(m.momentum.ρv))
        file["ρw"] = Array(Oceananigans.interior(m.momentum.ρw))
        file["ρθ"] = Array(Oceananigans.interior(m.formulation.potential_temperature_density))
        file["ρqᵛ"] = Array(Oceananigans.interior(m.moisture_density))
        for name in keys(m.microphysical_fields)
            file[string(name)] = Array(Oceananigans.interior(m.microphysical_fields[name]))
        end
    end
    @info "Saved rank $rank → $filepath"
end

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

Rx, Ry = 4, 2
@assert nprocs == Rx * Ry
arch = NCCLDistributed(GPU(); partition=Partition(Rx, Ry, 1))

Nλ, Nφ, Nz = 8640, 3840, 64
H = 30e3
sst_anomaly = 2.0
τ_cloud = 120

Δt_production = 0.8
total_sim_time = 6*3600.0  # 6h continuation
save_iter_interval = round(Int, 3600.0 / Δt_production)  # 1h cadence = 4500 iters

ic_path = joinpath(@__DIR__, "..", "simulations", "initial_conditions",
                   "twentyfourth_deg_iter9000_assembled.jld2")
isfile(ic_path) || error("IC not found: $ic_path")

rank == 0 && @info "Config" Nλ Nφ Nz Δt_production total_sim_time save_iter_interval ic_path

@time "build" model = build_model(arch; Nλ, Nφ, Nz, H, Δt=Δt_production, halo=(4,4,4),
                                   latitude=(-80,80), sst_anomaly, τ_cloud)

rank == 0 && @info "Loading IC..." now(UTC)
@time "load IC" load_ic_same_resolution!(model, ic_path; Rx, Ry)
any_nan(model) && error("NaN after IC load on rank $rank")
# Find where ρ=0 in the interior
let ρ_int = Array(Oceananigans.interior(dynamics_density(model.dynamics)))
    nzeros = count(==(0), ρ_int)
    ρ_min = minimum(ρ_int)
    if nzeros > 0 || ρ_min <= 0
        idxs = findall(x -> x <= 0, ρ_int)
        @warn @sprintf("[r%d] ρ interior has %d cells ≤ 0 (min=%.4e)", rank, length(idxs), ρ_min)
        for idx in idxs[1:min(5, length(idxs))]
            @warn @sprintf("[r%d]   ρ≤0 at (%d,%d,%d) = %.4e", rank, Tuple(idx)..., ρ_int[idx])
        end
    else
        @info @sprintf("[r%d] ρ interior OK: min=%.4e, no zeros", rank, ρ_min)
    end
end
rank == 0 && @info "IC loaded"

# ── Diagnostic: check state BETWEEN update_state! and time_step! ──────
rank == 0 && @info "Calling update_state!..." now(UTC)
Oceananigans.TimeSteppers.update_state!(model)

# Check all prognostic fields after update_state!
for (name, f) in [("ρ", dynamics_density(model.dynamics)),
                   ("ρu", model.momentum.ρu), ("ρv", model.momentum.ρv),
                   ("ρw", model.momentum.ρw),
                   ("ρθ", model.formulation.potential_temperature_density),
                   ("ρqᵛ", model.moisture_density)]
    mn, mx = Float64(minimum(Oceananigans.interior(f))), Float64(maximum(Oceananigans.interior(f)))
    @info @sprintf("[r%d] after update_state! %s interior: [%.4e, %.4e] finite=%s", rank, name, mn, mx, isfinite(mn)&&isfinite(mx))
end

# Check diagnostic/auxiliary fields
all_fields = Oceananigans.fields(model)
for name in keys(all_fields)
    f = all_fields[name]
    int = Oceananigans.interior(f)
    mn, mx = Float64(minimum(int)), Float64(maximum(int))
    fin = isfinite(mn) && isfinite(mx)
    if !fin
        @error @sprintf("[r%d] NaN/Inf in %s after update_state!: [%.4e, %.4e]", rank, name, mn, mx)
    else
        @info @sprintf("[r%d] %s: [%.4e, %.4e]", rank, name, mn, mx)
    end
end
flush(stderr); flush(stdout)
MPI.Barrier(MPI.COMM_WORLD)

any_nan(model) && @warn "NaN in prognostic fields after update_state! on rank $rank"

rank == 0 && @info "Calling time_step!(Δt=$Δt_production)..." now(UTC)
Oceananigans.TimeSteppers.time_step!(model, Δt_production)

# Check after time_step!
for (name, f) in [("ρ", dynamics_density(model.dynamics)),
                   ("ρu", model.momentum.ρu), ("ρv", model.momentum.ρv),
                   ("ρw", model.momentum.ρw),
                   ("ρθ", model.formulation.potential_temperature_density),
                   ("ρqᵛ", model.moisture_density)]
    mn, mx = Float64(minimum(Oceananigans.interior(f))), Float64(maximum(Oceananigans.interior(f)))
    fin = isfinite(mn) && isfinite(mx)
    if !fin
        @error @sprintf("[r%d] NaN/Inf in %s after time_step!: [%.4e, %.4e]", rank, name, mn, mx)
    end
end
any_nan(model) && error("NaN after first step on rank $rank")
rank == 0 && @info "First step OK" model.clock

# Output setup
output_dir = joinpath(@__DIR__, "..", "simulations", "output", "nccl_8gpu_24th_deg_continued")
rank == 0 && mkpath(output_dir)
MPI.Barrier(MPI.COMM_WORLD)
output_prefix = joinpath(output_dir, "fields_rank$rank")

# ── Production: 6h continuation, save every save_iter_interval iters ─
prod_iters = round(Int, total_sim_time / Δt_production)
final_iter = model.clock.iteration + prod_iters
rank == 0 && @info "Production: Δt=$Δt_production for $(total_sim_time)s ($prod_iters iters), save every $save_iter_interval iters" now(UTC)
sim = Simulation(model; Δt=Δt_production, stop_iteration=final_iter)
sim.callbacks[:diag] = Callback(diagnostics, IterationInterval(100))
sim.callbacks[:save] = Callback(
    sim -> save_output(sim, output_prefix, save_iter_interval),
    IterationInterval(save_iter_interval))
wall_start[] = time_ns()
Oceananigans.run!(sim)

rank == 0 && @info "Done!" now(UTC) model.clock
MPI.Finalize()
