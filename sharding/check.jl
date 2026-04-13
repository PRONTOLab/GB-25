#=
Multi-node interpolation correctness check.

Builds the actual AtmosphereModel (reusing moist_baroclinic_wave_model) and tests
that Reactant InterpolateArray nearest-neighbor matches the vanilla CPU kernel for
every prognostic field — including face fields with their different source sizes.

  Multi-node:  sbatch check_multinode.sh
  Single-node: julia --project=.. -O0 check.jl --grid-x 64 --grid-y 64 --grid-z 8
=#

using Dates
@info "Interpolation check starting" now(UTC)

ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using GordonBell25
using GordonBell25: factors, is_distributed_env_present
using Breeze.AtmosphereModels: dynamics_density
using Oceananigans
using Oceananigans.Architectures: ReactantState

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64,
    grid_y_default = 64,
    grid_z_default = 8,
)

Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)
FT = Oceananigans.defaults.FloatType

using Oceananigans.Units
using Printf
using Reactant
using Reactant: Sharding, InterpolateArray, InterpolationType

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.preamble()
GordonBell25.initialize(; single_gpu_per_process=false)

local_arch = ReactantState()
arch = local_arch

Ndev = length(Reactant.devices())
@show Ndev

Rx, Ry = factors(Ndev)

rank = 0
if Ndev > 1
    arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
    rank = Reactant.Distributed.local_rank()
end

@info "[$rank] Distributed setup" Ndev Rx Ry

# ─── Grid parameters ─────────────────────────────────────────────────────
H_halo = 4
Tλ = parsed_args["grid-x"] * Rx
Tφ = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]
Nλ = Tλ - 2H_halo
Nφ = Tφ - 2H_halo

column_height = 30e3
Δt = 0.5

@info "[$rank] Building model (Nλ=$Nλ, Nφ=$Nφ, Nz=$Nz)" now(UTC)

model = GordonBell25.moist_baroclinic_wave_model(arch; Nλ, Nφ, Nz,
    H = column_height, Δt,
    halo = (H_halo, H_halo, 4),
    cloud_formation_τ_relax = 10.0,
    initial_conditions_path = nothing)

@info "[$rank] Model built" now(UTC)

# ─── Deterministic source arrays mimicking the IC file ────────────────────
# Source grid is smaller than the target → upscaling, same as the real IC path
Nλ_src, Nφ_src, Nz_src = 32, 16, Nz

# Spatially varying patterns: each field gets a distinct structure so that
# any cross-field mixup or index transposition is immediately visible.
function make_src(nx, ny, nz; scale=FT(1), zscale=FT(1), offset=FT(0))
    arr = Array{FT}(undef, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        λ = FT(2π * (i - 1) / nx)              # zonal wave
        φ = FT(π * (j - 1) / ny - π/2)         # meridional: −π/2 … +π/2
        z = FT((k - 1) / max(nz - 1, 1))       # vertical: 0 … 1
        arr[i, j, k] = offset + scale * (sin(λ) * cos(φ) + zscale * z)
    end
    return arr
end

# Source sizes match the IC file convention:
#   center fields:  (Nλ_src, Nφ_src,   Nz_src  )
#   x-face (ρu):    (Nλ_src, Nφ_src,   Nz_src  )  ← same as center in the file
#   y-face (ρv):    (Nλ_src, Nφ_src+1, Nz_src  )
#   z-face (ρw):    (Nλ_src, Nφ_src,   Nz_src+1)
src_ρ    = make_src(Nλ_src, Nφ_src,     Nz_src;     scale=FT(1.2),  zscale=FT(-0.3), offset=FT(1.0))
src_ρu   = make_src(Nλ_src, Nφ_src,     Nz_src;     scale=FT(20),   zscale=FT(5),    offset=FT(0))
src_ρv   = make_src(Nλ_src, Nφ_src + 1, Nz_src;     scale=FT(10),   zscale=FT(3),    offset=FT(0))
src_ρw   = make_src(Nλ_src, Nφ_src,     Nz_src + 1; scale=FT(0.5),  zscale=FT(0.2),  offset=FT(0))
src_ρθ   = make_src(Nλ_src, Nφ_src,     Nz_src;     scale=FT(15),   zscale=FT(-10),  offset=FT(330))
src_ρqv  = make_src(Nλ_src, Nφ_src,     Nz_src;     scale=FT(1e-4), zscale=FT(-5e-5),offset=FT(3e-4))
src_ρqcl = make_src(Nλ_src, Nφ_src,     Nz_src;     scale=FT(1e-6), zscale=FT(2e-6), offset=FT(1e-6))
src_ρqci = make_src(Nλ_src, Nφ_src,     Nz_src;     scale=FT(3e-4), zscale=FT(2e-4), offset=FT(5e-4))

field_pairs = [
    ("ρ",    src_ρ,    dynamics_density(model.dynamics)),
    ("ρu",   src_ρu,   model.momentum.ρu),
    ("ρv",   src_ρv,   model.momentum.ρv),
    ("ρw",   src_ρw,   model.momentum.ρw),
    ("ρθ",   src_ρθ,   model.formulation.potential_temperature_density),
    ("ρqᵛ",  src_ρqv,  model.moisture_density),
    ("ρqᶜˡ", src_ρqcl, model.microphysical_fields[:ρqᶜˡ]),
    ("ρqᶜⁱ", src_ρqci, model.microphysical_fields[:ρqᶜⁱ]),
]

# ═════════════════════════════════════════════════════════════════════════
# Path 1: Reactant InterpolateArray (as in set_moist_baroclinic_wave_from_file!)
# ═════════════════════════════════════════════════════════════════════════

grid = model.grid
halo = Oceananigans.halo_size(grid)

@info "[$rank] Running Reactant InterpolateArray for all fields..." now(UTC)

reactant_interiors = Dict{String, Array{FT}}()

for (name, src_array, target_field) in field_pairs
    target_data = Reactant.ancestor(target_field)
    target_size = size(target_data)
    target_sharding = target_data.sharding

    @info "[$rank] InterpolateArray" name src=size(src_array) dst=target_size halo

    # Pass halo=(0,0,0): source has no halos (raw IC data), matching production code.
    result = InterpolateArray(FT.(src_array), target_size, target_sharding,
                              InterpolationType.Nearest, ntuple(_ -> 0, 3))

    full_arr = Array(result)

    # Extract interior (strip halos)
    Hx, Hy, Hz = halo
    interior_size = size(Oceananigans.interior(target_field))
    Nx, Ny, Nzf = interior_size
    interior = full_arr[(Hx+1):(Hx+Nx), (Hy+1):(Hy+Ny), (Hz+1):(Hz+Nzf)]
    reactant_interiors[name] = interior
end

# ═════════════════════════════════════════════════════════════════════════
# Path 2: CPU nearest-neighbor (as in set_moist_baroclinic_wave_from_file_vanilla!)
# ═════════════════════════════════════════════════════════════════════════

@info "[$rank] Running CPU nearest-neighbor for all fields..." now(UTC)

cpu_interiors = Dict{String, Array{FT}}()

# With halo=(0,0,0), Reactant's nearest-neighbor mapping reduces to:
#   idx = (I * M_dim + N_dim - 1) ÷ N_dim    clamped to [1, M_dim]
# where I is the 1-based index in the full target (including halos) and
# N_dim/M_dim are the full target/source sizes.
Hx, Hy, Hz = halo

for (name, src_array, target_field) in field_pairs
    target_data = Reactant.ancestor(target_field)
    N_full = size(target_data)   # full target size including halos
    M_full = size(src_array)     # source size (no halos)

    Nx_int, Ny_int, Nz_int = size(Oceananigans.interior(target_field))

    cpu_arr = zeros(FT, Nx_int, Ny_int, Nz_int)
    for k in 1:Nz_int, j in 1:Ny_int, i in 1:Nx_int
        I = i + Hx;  J = j + Hy;  K = k + Hz
        i′ = clamp((I * M_full[1] + N_full[1] - 1) ÷ N_full[1], 1, M_full[1])
        j′ = clamp((J * M_full[2] + N_full[2] - 1) ÷ N_full[2], 1, M_full[2])
        k′ = clamp((K * M_full[3] + N_full[3] - 1) ÷ N_full[3], 1, M_full[3])
        cpu_arr[i, j, k] = src_array[i′, j′, k′]
    end
    cpu_interiors[name] = cpu_arr

    @info "[$rank] CPU ref" name src=M_full dst=N_full interior=(Nx_int, Ny_int, Nz_int) halo
end

# ═════════════════════════════════════════════════════════════════════════
# Compare
# ═════════════════════════════════════════════════════════════════════════

@info "[$rank] Comparing results..." now(UTC)

function compare_region(rank, name, r, c, region_name, i_range, j_range, k_range)
    rv = @view r[i_range, j_range, k_range]
    cv = @view c[i_range, j_range, k_range]
    diff = abs.(rv .- cv)
    n_mis = count(diff .> 0)
    max_diff = maximum(diff)
    n_total = length(diff)

    if n_mis > 0
        @warn "[$rank] FAIL $name ($region_name)" max_abs_diff=max_diff n_mismatch=n_mis n_total
        reported = 0
        for k in axes(diff,3), j in axes(diff,2), i in axes(diff,1)
            if diff[i,j,k] > 0
                reported += 1
                gi = i_range[i]; gj = j_range[j]; gk = k_range[k]
                @info "  mismatch" name region=region_name idx=(gi,gj,gk) reactant=rv[i,j,k] cpu=cv[i,j,k] delta=diff[i,j,k]
                reported >= 5 && break
            end
        end
        return false
    else
        @info "[$rank] PASS $name ($region_name)" n_total range=extrema(rv)
        return true
    end
end

all_pass = true

for (name, _, _) in field_pairs
    r = reactant_interiors[name]
    c = cpu_interiors[name]
    Ni, Nj, Nk = size(r)

    # Full interior
    pass_full = compare_region(rank, name, r, c, "full",
        1:Ni, 1:Nj, 1:Nk)

    # Border strip (first & last 2 rows/cols in x and y)
    b = 2
    pass_border = compare_region(rank, name, r, c, "border",
        vcat(1:b, (Ni-b+1):Ni), vcat(1:b, (Nj-b+1):Nj), 1:Nk)

    # Deep interior (skip 10% from each edge)
    mx = max(1, Ni ÷ 10); my = max(1, Nj ÷ 10); mz = max(1, Nk ÷ 4)
    pass_deep = compare_region(rank, name, r, c, "deep interior",
        (mx+1):(Ni-mx), (my+1):(Nj-my), (mz+1):(Nk-mz))

    # Single center point
    ci, cj, ck = Ni÷2, Nj÷2, max(1, Nk÷2)
    rv = r[ci, cj, ck]; cv = c[ci, cj, ck]
    @info "[$rank] $name center ($ci,$cj,$ck)" reactant=rv cpu=cv match=(rv==cv)

    global all_pass = all_pass & pass_full & pass_border & pass_deep
end

if all_pass
    @info "[$rank] ALL NEAREST-NEIGHBOR FIELDS PASS"
else
    @error "[$rank] SOME NEAREST-NEIGHBOR FIELDS FAILED"
end

# ═════════════════════════════════════════════════════════════════════════
# Part 2: Linear interpolation — Reactant vs CPU trilinear reference
# ═════════════════════════════════════════════════════════════════════════

@info "[$rank] ═══ LINEAR INTERPOLATION TEST ═══" now(UTC)

const LINEAR_TOL = FT(1e-7)

# --- Reactant linear path ---

@info "[$rank] Running Reactant InterpolateArray (Linear) for all fields..." now(UTC)

reactant_linear = Dict{String, Array{FT}}()

for (name, src_array, target_field) in field_pairs
    target_data = Reactant.ancestor(target_field)
    target_size = size(target_data)
    target_sharding = target_data.sharding

    result = InterpolateArray(FT.(src_array), target_size, target_sharding,
                              InterpolationType.Linear, ntuple(_ -> 0, 3))

    full_arr = Array(result)

    Hx, Hy, Hz = halo
    Nx, Ny, Nzf = size(Oceananigans.interior(target_field))
    interior = full_arr[(Hx+1):(Hx+Nx), (Hy+1):(Hy+Ny), (Hz+1):(Hz+Nzf)]
    reactant_linear[name] = interior
end

# --- CPU trilinear reference (cell-centered, matching Reactant's formula) ---
# With halo=(0,0,0), for target index I in [1, N]:
#   a = (2I - 1) * M + N
#   b = 2N
#   low  = clamp(a ÷ b,     1, M)
#   high = clamp(a ÷ b + 1, 1, M)
#   rem  = a % b
#   weight_low  = (b - rem) / b
#   weight_high = rem / b
# Trilinear: sum over 8 corners of (product of weights) * src[corner]

@info "[$rank] Running CPU trilinear for all fields..." now(UTC)

cpu_linear = Dict{String, Array{FT}}()

for (name, src_array, target_field) in field_pairs
    target_data = Reactant.ancestor(target_field)
    N_full = size(target_data)
    M_full = size(src_array)

    Nx_int, Ny_int, Nz_int = size(Oceananigans.interior(target_field))
    cpu_arr = zeros(FT, Nx_int, Ny_int, Nz_int)

    dens = ntuple(d -> 2 * N_full[d], 3)
    total_den = FT(prod(dens))

    for k in 1:Nz_int, j in 1:Ny_int, i in 1:Nx_int
        Is = (i + Hx, j + Hy, k + Hz)

        a = ntuple(d -> (2 * Is[d] - 1) * M_full[d] + N_full[d], 3)
        lo = ntuple(d -> clamp(a[d] ÷ dens[d],     1, M_full[d]), 3)
        hi = ntuple(d -> clamp(a[d] ÷ dens[d] + 1, 1, M_full[d]), 3)
        r  = ntuple(d -> a[d] % dens[d], 3)

        val = zero(FT)
        for cz in 0:1, cy in 0:1, cx in 0:1
            ix = cx == 0 ? lo[1] : hi[1]
            iy = cy == 0 ? lo[2] : hi[2]
            iz = cz == 0 ? lo[3] : hi[3]
            w = FT(1)
            w *= cx == 0 ? (dens[1] - r[1]) : r[1]
            w *= cy == 0 ? (dens[2] - r[2]) : r[2]
            w *= cz == 0 ? (dens[3] - r[3]) : r[3]
            val += w * src_array[ix, iy, iz]
        end
        cpu_arr[i, j, k] = val / total_den
    end
    cpu_linear[name] = cpu_arr

    @info "[$rank] CPU trilinear" name src=M_full dst=N_full interior=(Nx_int, Ny_int, Nz_int)
end

# --- Compare with tolerance ---

function compare_linear(rank, name, r, c, tol)
    diff = abs.(r .- c)
    max_diff = maximum(diff)
    n_mis = count(diff .> tol)
    n_total = length(diff)

    if n_mis > 0
        @warn "[$rank] LINEAR FAIL $name" max_abs_diff=max_diff n_exceed_tol=n_mis n_total tol
        reported = 0
        for k in axes(diff,3), j in axes(diff,2), i in axes(diff,1)
            if diff[i,j,k] > tol
                reported += 1
                @info "  linear mismatch" name idx=(i,j,k) reactant=r[i,j,k] cpu=c[i,j,k] delta=diff[i,j,k]
                reported >= 5 && break
            end
        end
        return false
    else
        @info "[$rank] LINEAR PASS $name" max_abs_diff=max_diff n_total tol range=extrema(r)
        return true
    end
end

all_linear_pass = true

for (name, _, _) in field_pairs
    r = reactant_linear[name]
    c = cpu_linear[name]
    pass = compare_linear(rank, name, r, c, LINEAR_TOL)
    global all_linear_pass = all_linear_pass & pass
end

if all_linear_pass
    @info "[$rank] ALL LINEAR FIELDS PASS (tol=$LINEAR_TOL)"
else
    @error "[$rank] SOME LINEAR FIELDS FAILED (tol=$LINEAR_TOL)"
end

# ═════════════════════════════════════════════════════════════════════════
# Final summary
# ═════════════════════════════════════════════════════════════════════════

if all_pass && all_linear_pass
    @info "[$rank] ✓ ALL TESTS PASS (nearest + linear)"
else
    @error "[$rank] TESTS FAILED" nearest=all_pass linear=all_linear_pass
end
