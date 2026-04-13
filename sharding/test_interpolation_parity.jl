#!/usr/bin/env julia
#
# Test: Reactant InterpolateArray path vs vanilla Oceananigans interpolate! path
#
# Compares the two IC-loading pipelines defined in moist_baroclinic_wave_model.jl:
#
#   Reactant  (set_moist_baroclinic_wave_from_file!):
#     CPU Field → fill_halo_regions! → trim face+bounded → InterpolateArray
#
#   Vanilla   (set_moist_baroclinic_wave_from_file_vanilla!):
#     CPU Field → fill_halo_regions! → Oceananigans.Fields.interpolate!
#
# Tests every staggering: (C,C,C), (F,C,C), (C,F,C), (C,C,F).
#
# Part 1: small random data  (32×16×8 → 32×16×8,  same-size identity)
# Part 2: small random data  (32×16×8 → 16×8×4,   downscale)
# Part 3: real IC data        (1536×768×64 → 1536×768×64,  same-size)
#
# Usage:
#   XLA_FLAGS="--xla_force_host_platform_device_count=1" \
#     julia --project=. sharding/test_interpolation_parity.jl

ENV["XLA_FLAGS"] = get(ENV, "XLA_FLAGS", "") *
                   " --xla_force_host_platform_device_count=1"

using Printf, Random, Dates
@info "Interpolation parity test" now(UTC)

using Oceananigans
using Oceananigans.Grids: Center, Face, Bounded
using Oceananigans.Fields: CenterField, XFaceField, YFaceField, ZFaceField, Field
using Oceananigans.BoundaryConditions: fill_halo_regions!
Oceananigans.defaults.FloatType = Float32

using Reactant
using Reactant: InterpolateArray, InterpolationType, Sharding
using JLD2
using GordonBell25
using GordonBell25: FaceInterpolateArray, prepare_source_for_interpolation

mesh     = Sharding.Mesh(reshape(Reactant.devices()[1:1], 1), (:x,))
sharding = Sharding.NamedSharding(mesh, ("x", nothing, nothing))

const FT = Float32

# ═══════════════════════════════════════════════════════════════════════════════
# Shared: build a CPU source field from raw data, fill halos
# (both paths use this — mirrors moist_baroclinic_wave_model.jl)
# ═══════════════════════════════════════════════════════════════════════════════

function make_source_field(src_data, loc, src_grid)
    iloc = map(L -> L(), loc)
    src_field = Field(iloc, src_grid)
    Oceananigans.interior(src_field) .= FT.(src_data)
    fill_halo_regions!(src_field)
    return src_field
end

# ═══════════════════════════════════════════════════════════════════════════════
# Reactant path:  prepare_source → InterpolateArray (center) or
#                                  FaceInterpolateArray (face)
# (mirrors set_moist_baroclinic_wave_from_file! hybrid approach)
# ═══════════════════════════════════════════════════════════════════════════════

const BOUNDED_DIMS = (false, true, true)  # LatitudeLongitudeGrid: Periodic, Bounded, Bounded

function reactant_interpolate(src_data, loc, target_size, halo, sharding;
                              interpolation = InterpolationType.Linear)
    face_dims = ntuple(d -> loc[d] === Face, 3)
    is_face = any(face_dims)

    padded = prepare_source_for_interpolation(src_data, halo, face_dims, FT)

    result = if is_face
        FaceInterpolateArray(padded, target_size, sharding,
                             interpolation, halo; face_dims,
                             bounded_dims=BOUNDED_DIMS)
    else
        InterpolateArray(padded, target_size, sharding, interpolation, halo)
    end
    return Array(result), padded
end

# ═══════════════════════════════════════════════════════════════════════════════
# Vanilla path:  interpolate!(target_field, src_field)
# (mirrors set_moist_baroclinic_wave_from_file_vanilla!, lines 641-650)
# ═══════════════════════════════════════════════════════════════════════════════

function vanilla_interpolate(src_field, tgt_grid)
    loc  = Oceananigans.Fields.location(src_field)
    iloc = map(L -> L(), loc)
    tgt_field = Field(iloc, tgt_grid)
    Oceananigans.Fields.interpolate!(tgt_field, src_field)
    return tgt_field
end

# ═══════════════════════════════════════════════════════════════════════════════
# Compare one field
# ═══════════════════════════════════════════════════════════════════════════════

function compare_field(name, src_data, loc, src_grid, tgt_grid, halo, sharding)
    Hx, Hy, Hz = halo
    face_dims = (loc[1] === Face, loc[2] === Face, loc[3] === Face)

    # ─── Source field for vanilla path ────────────────────────────────────
    src_field = make_source_field(src_data, loc, src_grid)

    # ─── Reactant path (prepare_source → FaceInterpolateArray/InterpolateArray) ──
    Nλ_t, Nφ_t, Nz_t = size(tgt_grid)
    tgt_parent_size = (Nλ_t + 2Hx, Nφ_t + 2Hy, Nz_t + 2Hz)
    reactant_full, padded = reactant_interpolate(src_data, loc, tgt_parent_size,
                                                  halo, sharding)
    ri = reactant_full[Hx+1:end-Hx, Hy+1:end-Hy, Hz+1:end-Hz]

    # ─── Vanilla path (Oceananigans interpolate!) ─────────────────────────
    tgt_field = vanilla_interpolate(src_field, tgt_grid)
    vanilla_int = Array(Oceananigans.interior(tgt_field))

    # Trim vanilla interior to match Reactant convention for face+bounded
    vi = vanilla_int
    if face_dims[2]; vi = vi[:, 1:end-1, :]; end
    if face_dims[3]; vi = vi[:, :, 1:end-1]; end

    # ─── Metrics ─────────────────────────────────────────────────────────
    diff_v_r = maximum(abs.(vi .- ri))
    scale    = max(maximum(abs, vi), one(FT))
    rel_diff = diff_v_r / scale

    # Source data fidelity (only meaningful for same-size grids)
    st = FT.(src_data)
    if face_dims[2]; st = st[:, 1:end-1, :]; end
    if face_dims[3]; st = st[:, :, 1:end-1]; end

    same_size = size(st) == size(vi)
    diff_s_v = same_size ? maximum(abs.(st .- vi)) : NaN32
    diff_s_r = same_size ? maximum(abs.(st .- ri)) : NaN32

    pass = rel_diff < 1e-4

    if same_size
        @printf("  %-16s │ van↔rct %9.2e (rel %7.1e) │ src↔van %9.2e │ src↔rct %9.2e │ %s\n",
                name, diff_v_r, rel_diff, diff_s_v, diff_s_r,
                pass ? "✓ PASS" : "✗ FAIL")
    else
        @printf("  %-16s │ van↔rct %9.2e (rel %7.1e) │ src(%s) → tgt(%s) │ %s\n",
                name, diff_v_r, rel_diff,
                join(size(st), "×"), join(size(ri), "×"),
                pass ? "✓ PASS" : "✗ FAIL")
    end
    @printf("                   │ src  (%+11.4e, %+11.4e)  van  (%+11.4e, %+11.4e)  rct  (%+11.4e, %+11.4e)\n",
            extrema(st)..., extrema(vi)..., extrema(ri)...)

    if !pass
        idx = argmax(abs.(vi .- ri))
        @warn "  Worst mismatch at $idx" vanilla=vi[idx] reactant=ri[idx] delta=vi[idx]-ri[idx]
    end

    return (; pass, diff_v_r, rel_diff, diff_s_v, diff_s_r)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run a batch of fields
# ═══════════════════════════════════════════════════════════════════════════════

function run_batch(title, specs, src_grid, tgt_grid, halo, sharding)
    println("\n", "=" ^ 120)
    println(title)
    println("=" ^ 120)
    println()

    results = [compare_field(n, d, loc, src_grid, tgt_grid, halo, sharding)
               for (n, d, loc) in specs]
    println()

    ok = all(r -> r.pass, results)
    println("  Result: ", ok ? "ALL PASS ✓" : "SOME FAILED ✗")
    return ok
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — Same-size, small random data
# ═══════════════════════════════════════════════════════════════════════════════

function run_part1()
    Random.seed!(42)
    Nλ, Nφ, Nz = 32, 16, 8
    H = (4, 4, 4)

    src_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ, Nφ, Nz), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))
    tgt_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ, Nφ, Nz), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))

    specs = [
        ("ρ  (C,C,C)",  randn(FT, Nλ,   Nφ,     Nz    ), (Center, Center, Center)),
        ("ρu (F,C,C)",  randn(FT, Nλ,   Nφ,     Nz    ), (Face,   Center, Center)),
        ("ρv (C,F,C)",  randn(FT, Nλ,   Nφ + 1, Nz    ), (Center, Face,   Center)),
        ("ρw (C,C,F)",  randn(FT, Nλ,   Nφ,     Nz + 1), (Center, Center, Face  )),
    ]

    run_batch("PART 1  ·  Same-size random (32×16×8 → 32×16×8), halo=(4,4,4)",
              specs, src_grid, tgt_grid, H, sharding)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — Different-size, small random data (downscale)
# ═══════════════════════════════════════════════════════════════════════════════

function run_part2()
    Random.seed!(123)
    Nλ_s, Nφ_s, Nz_s = 32, 16, 8
    Nλ_t, Nφ_t, Nz_t = 16, 8, 4
    H = (4, 4, 4)

    src_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ_s, Nφ_s, Nz_s), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))
    tgt_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ_t, Nφ_t, Nz_t), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))

    specs = [
        ("ρ  (C,C,C)",  randn(FT, Nλ_s,   Nφ_s,     Nz_s    ), (Center, Center, Center)),
        ("ρu (F,C,C)",  randn(FT, Nλ_s,   Nφ_s,     Nz_s    ), (Face,   Center, Center)),
        ("ρv (C,F,C)",  randn(FT, Nλ_s,   Nφ_s + 1, Nz_s    ), (Center, Face,   Center)),
        ("ρw (C,C,F)",  randn(FT, Nλ_s,   Nφ_s,     Nz_s + 1), (Center, Center, Face  )),
    ]

    run_batch("PART 2  ·  Downscale random (32×16×8 → 16×8×4), halo=(4,4,4)",
              specs, src_grid, tgt_grid, H, sharding)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — Real IC data, same-size (1536×768×64)
# ═══════════════════════════════════════════════════════════════════════════════

function run_part3()
    ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                       "atmosphere_coarsened_1536x768x64.jld2")

    if !isfile(ic_path)
        @warn "IC file not found — skipping Part 3" ic_path
        return nothing
    end

    Nλ_s, Nφ_s, Nz_s, ρ_d, ρu_d, ρv_d, ρw_d, ρθ_d, ρqv_d =
        JLD2.jldopen(ic_path, "r") do f
            (f["Nλ"], f["Nφ"], f["Nz"],
             f["ρ"], f["ρu"], f["ρv"], f["ρw"],
             f["ρθ"], f["ρqᵛ"])
        end

    @info "IC source" Nλ=Nλ_s Nφ=Nφ_s Nz=Nz_s
    @info "Raw extrema" ρ=extrema(ρ_d) ρu=extrema(ρu_d) ρv=extrema(ρv_d) ρw=extrema(ρw_d) ρθ=extrema(ρθ_d) ρqᵛ=extrema(ρqv_d)

    H = (4, 4, 4)

    src_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ_s, Nφ_s, Nz_s), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))
    tgt_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ_s, Nφ_s, Nz_s), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))

    specs = [
        ("ρ   (C,C,C)",  ρ_d,   (Center, Center, Center)),
        ("ρu  (F,C,C)",  ρu_d,  (Face,   Center, Center)),
        ("ρv  (C,F,C)",  ρv_d,  (Center, Face,   Center)),
        ("ρw  (C,C,F)",  ρw_d,  (Center, Center, Face  )),
        ("ρθ  (C,C,C)",  ρθ_d,  (Center, Center, Center)),
        ("ρqᵛ (C,C,C)",  ρqv_d, (Center, Center, Center)),
    ]

    ok = run_batch("PART 3  ·  Real IC same-size (1536×768×64 → 1536×768×64), halo=(4,4,4)",
                   specs, src_grid, tgt_grid, H, sharding)

    println("\n  Density zero-check (interior should never be exactly 0):")
    for (n, d, _) in specs
        if any(contains(n, p) for p in ("ρ ", "ρθ"))
            td = FT.(d)
            @printf("    %-14s min=%+.4e  has_zero=%s\n", n, minimum(td), any(==(0), td))
        end
    end

    return ok
end

# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — Real IC data, upscale (1536×768×64 → 1920×960×64)
# ═══════════════════════════════════════════════════════════════════════════════

function run_part4()
    ic_path = joinpath(pkgdir(GordonBell25), "simulations", "initial_conditions",
                       "atmosphere_coarsened_1536x768x64.jld2")

    if !isfile(ic_path)
        @warn "IC file not found — skipping Part 4" ic_path
        return nothing
    end

    Nλ_s, Nφ_s, Nz_s, ρ_d, ρu_d, ρv_d, ρw_d, ρθ_d, ρqv_d =
        JLD2.jldopen(ic_path, "r") do f
            (f["Nλ"], f["Nφ"], f["Nz"],
             f["ρ"], f["ρu"], f["ρv"], f["ρw"],
             f["ρθ"], f["ρqᵛ"])
        end

    Nλ_t, Nφ_t, Nz_t = 1920, 960, 64
    @info "IC upscale" src=(Nλ_s, Nφ_s, Nz_s) tgt=(Nλ_t, Nφ_t, Nz_t)

    H = (4, 4, 4)

    src_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ_s, Nφ_s, Nz_s), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))
    tgt_grid = LatitudeLongitudeGrid(CPU(), FT;
        size=(Nλ_t, Nφ_t, Nz_t), halo=H,
        longitude=(0, 360), latitude=(-80, 80), z=(0, 30e3))

    specs = [
        ("ρ   (C,C,C)",  ρ_d,   (Center, Center, Center)),
        ("ρu  (F,C,C)",  ρu_d,  (Face,   Center, Center)),
        ("ρv  (C,F,C)",  ρv_d,  (Center, Face,   Center)),
        ("ρw  (C,C,F)",  ρw_d,  (Center, Center, Face  )),
        ("ρθ  (C,C,C)",  ρθ_d,  (Center, Center, Center)),
        ("ρqᵛ (C,C,C)",  ρqv_d, (Center, Center, Center)),
    ]

    ok = run_batch("PART 4  ·  Real IC upscale (1536×768×64 → 1920×960×64), halo=(4,4,4)",
                   specs, src_grid, tgt_grid, H, sharding)

    println("\n  Density zero-check (target interior should never be exactly 0):")
    for (n, d, loc) in specs
        if any(contains(n, p) for p in ("ρ ", "ρθ"))
            tgt_sz = (Nλ_t + 2H[1], Nφ_t + 2H[2], Nz_t + 2H[3])
            rct_full, _ = reactant_interpolate(d, loc, tgt_sz, H, sharding)
            ri = rct_full[H[1]+1:end-H[1], H[2]+1:end-H[2], H[3]+1:end-H[3]]
            @printf("    %-14s rct_min=%+.4e  has_zero=%s\n", n, minimum(ri), any(==(0), ri))
        end
    end

    return ok
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

ok1 = run_part1()
ok2 = run_part2()
ok3 = run_part3()
ok4 = run_part4()

println("\n", "=" ^ 120)
results = [ok1, ok2, ok3, ok4]
overall = all(r -> r !== false, results)
println(overall ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗")
println("=" ^ 120)
