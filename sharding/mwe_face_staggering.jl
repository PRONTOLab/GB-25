#!/usr/bin/env julia
#
# MWE: InterpolateArray half-cell registration error for Face fields.
#
# Two distinct problems exist when using InterpolateArray for staggered-grid
# atmospheric data:
#
#   1. BOUNDARY CLAMPING — InterpolateArray can't wrap around periodic
#      boundaries, so points near λ=0°/360° get clamped edge values instead
#      of proper periodic interpolation.  Affects center AND face fields.
#
#   2. STAGGERING OFFSET — InterpolateArray's half-pixel convention assumes
#      data lives at cell centres.  For face fields the nodes are offset by
#      Δλ/2, so the interpolation samples at the wrong physical locations.
#      This is a systematic error everywhere in the domain, not just at
#      boundaries.
#
# This script quantifies both effects by comparing InterpolateArray output
# against (a) analytic truth and (b) a CPU linear interpolation that uses
# the correct face-to-face index mapping with periodic wrapping.
#
# Run:
#   XLA_FLAGS="--xla_force_host_platform_device_count=1" \
#     julia --project=. sharding/mwe_face_staggering.jl

ENV["XLA_FLAGS"] = get(ENV, "XLA_FLAGS", "") *
                   " --xla_force_host_platform_device_count=1"

using Printf
using Reactant
using Reactant: InterpolateArray, InterpolationType, Sharding
using GordonBell25: FaceInterpolateArray

mesh     = Sharding.Mesh(reshape(Reactant.devices()[1:1], 1), (:x,))
sharding = Sharding.NamedSharding(mesh, ("x", nothing, nothing))

# ── Grid sizes ────────────────────────────────────────────────────────────
Nλ_src = 16          # source longitude points
Nλ_tgt = 48          # target longitude points (3× upsample)
Δλ_src = 360f0 / Nλ_src
Δλ_tgt = 360f0 / Nλ_tgt

f(λ) = sinpi(2f0 * λ / 360f0)

# ── Source arrays (Nλ × 1 × 1) ───────────────────────────────────────────

center_src = Float32[f((i - 0.5f0) * Δλ_src) for i in 1:Nλ_src]
face_src   = Float32[f((i - 1f0)   * Δλ_src) for i in 1:Nλ_src]

center_src_3d = reshape(center_src, Nλ_src, 1, 1)
face_src_3d   = reshape(face_src,   Nλ_src, 1, 1)

# ── InterpolateArray resize (linear, no halo) ────────────────────────────

tgt_sz    = (Nλ_tgt, 1, 1)
center_ia = vec(Array(InterpolateArray(center_src_3d, tgt_sz, sharding,
                                       InterpolationType.Linear, (0, 0, 0))))
face_ia   = vec(Array(InterpolateArray(face_src_3d,   tgt_sz, sharding,
                                       InterpolationType.Linear, (0, 0, 0))))

# ── FaceInterpolateArray resize (linear, face_dims correction) ────────────

face_fia = vec(Array(FaceInterpolateArray(face_src_3d, tgt_sz, sharding,
                                          InterpolationType.Linear, (0, 0, 0);
                                          face_dims=(true, false, false))))

# ── Analytic truth at target locations ────────────────────────────────────

center_truth = Float32[f((i - 0.5f0) * Δλ_tgt) for i in 1:Nλ_tgt]
face_truth   = Float32[f((i - 1f0)   * Δλ_tgt) for i in 1:Nλ_tgt]

# ── CPU linear interpolation with PERIODIC wrapping ──────────────────────

function cpu_linear_periodic(src::Vector{Float32}, Ntgt, fi_func)
    Nsrc = length(src)
    dst  = zeros(Float32, Ntgt)
    for i in 1:Ntgt
        fi = fi_func(Float32(i), Float32(Nsrc), Float32(Ntgt))
        i⁻ = floor(Int, fi)
        ξ  = fi - i⁻
        i⁻ = mod1(i⁻, Nsrc)
        i⁺ = mod1(i⁻ + 1, Nsrc)    # wraps at periodic boundary
        dst[i] = (1f0 - ξ) * src[i⁻] + ξ * src[i⁺]
    end
    return dst
end

# Center-to-center fractional index (matches InterpolateArray convention):
fi_center(i, Ns, Nt) = (i - 0.5f0) * Ns / Nt + 0.5f0

# Face-to-face fractional index (correct staggered mapping):
fi_face(i, Ns, Nt) = (i - 1f0) * Ns / Nt + 1f0

center_cpu = cpu_linear_periodic(center_src, Nλ_tgt, fi_center)
face_cpu   = cpu_linear_periodic(face_src,   Nλ_tgt, fi_face)

# ── Error computation ─────────────────────────────────────────────────────

center_ia_err  = center_ia  .- center_truth
face_ia_err    = face_ia    .- face_truth
face_fia_err   = face_fia   .- face_truth     # FaceInterpolateArray (the fix)
face_cpu_err   = face_cpu   .- face_truth
stagger_err    = face_ia    .- face_cpu
center_cpu_err = center_cpu .- center_truth

# Interior: skip first and last Nλ_tgt÷Nλ_src points to avoid boundary region
margin  = Nλ_tgt ÷ Nλ_src
intslice = (margin+1):(Nλ_tgt - margin)

# ── Print: CenterField ────────────────────────────────────────────────────

println("=" ^ 90)
println("CenterField: InterpolateArray vs CPU-periodic vs truth   (Nsrc=$Nλ_src → Ntgt=$Nλ_tgt)")
println("=" ^ 90)
@printf("  Max |IA − truth|      all: %.6f   interior: %.6f\n",
        maximum(abs, center_ia_err), maximum(abs, center_ia_err[intslice]))
@printf("  Max |CPU − truth|     all: %.6f   interior: %.6f   (periodic wrapping)\n",
        maximum(abs, center_cpu_err), maximum(abs, center_cpu_err[intslice]))
println()
@printf("  %3s  %7s  %10s  %10s  %10s  %10s  %10s\n",
        "i", "λ_c(°)", "truth", "IA", "IA_err", "CPU_per", "CPU_err")
@printf("  %3s  %7s  %10s  %10s  %10s  %10s  %10s\n",
        "---", "-------", "----------", "----------", "----------",
        "----------", "----------")
for i in 1:Nλ_tgt
    λ = (i - 0.5f0) * Δλ_tgt
    @printf("  %3d  %7.1f  %+10.6f  %+10.6f  %+10.6f  %+10.6f  %+10.6f\n",
            i, λ, center_truth[i], center_ia[i], center_ia_err[i],
            center_cpu[i], center_cpu_err[i])
end

# ── Print: FaceField ──────────────────────────────────────────────────────

println()
println("=" ^ 100)
println("FaceField (XFace): IA vs FaceIA (FIX) vs CPU-periodic-face vs truth")
println("=" ^ 100)
@printf("  Max |IA − truth|        all: %.6f   interior: %.6f   (BROKEN — stagger offset)\n",
        maximum(abs, face_ia_err), maximum(abs, face_ia_err[intslice]))
@printf("  Max |FIA − truth|       all: %.6f   interior: %.6f   (FIX — face_dims correction)\n",
        maximum(abs, face_fia_err), maximum(abs, face_fia_err[intslice]))
@printf("  Max |CPU_face − truth|  all: %.6f   interior: %.6f   (periodic wrapping ref)\n",
        maximum(abs, face_cpu_err), maximum(abs, face_cpu_err[intslice]))
println()
@printf("  %3s  %7s  %10s  %10s  %10s  %10s  %10s  %10s\n",
        "i", "λ_f(°)", "truth", "IA", "IA_err", "FIA", "FIA_err", "CPU_face")
@printf("  %3s  %7s  %10s  %10s  %10s  %10s  %10s  %10s\n",
        "---", "-------", "----------", "----------", "----------",
        "----------", "----------", "----------")
for i in 1:Nλ_tgt
    λ = (i - 1f0) * Δλ_tgt
    @printf("  %3d  %7.1f  %+10.6f  %+10.6f  %+10.6f  %+10.6f  %+10.6f  %+10.6f\n",
            i, λ, face_truth[i], face_ia[i], face_ia_err[i],
            face_fia[i], face_fia_err[i], face_cpu[i])
end

# ── Summary ───────────────────────────────────────────────────────────────

println()
println("=" ^ 100)
println("Summary  (interior = indices $(first(intslice)):$(last(intslice)), avoiding boundary cells)")
println("=" ^ 100)

R = Float32(Nλ_src) / Float32(Nλ_tgt)
offset_cells = 0.5f0 * (R - 1f0)
offset_deg   = abs(offset_cells * Δλ_src)

@printf("\n  Systematic face offset = %.4f source cells = %.2f°\n", offset_cells, offset_deg)
println()
@printf("                                        All        Interior\n")
@printf("  CenterField  |IA − truth|           %8.6f     %8.6f    (boundary clamp only)\n",
        maximum(abs, center_ia_err), maximum(abs, center_ia_err[intslice]))
@printf("  CenterField  |CPU − truth|          %8.6f     %8.6f    (periodic → ≈ 0)\n",
        maximum(abs, center_cpu_err), maximum(abs, center_cpu_err[intslice]))
@printf("  FaceField    |IA − truth|           %8.6f     %8.6f    (BROKEN — stagger)\n",
        maximum(abs, face_ia_err), maximum(abs, face_ia_err[intslice]))
@printf("  FaceField    |FIA − truth|          %8.6f     %8.6f    ← FIX (face_dims)\n",
        maximum(abs, face_fia_err), maximum(abs, face_fia_err[intslice]))
@printf("  FaceField    |CPU − truth|          %8.6f     %8.6f    (periodic ref)\n",
        maximum(abs, face_cpu_err), maximum(abs, face_cpu_err[intslice]))
@printf("  FaceField    |IA − CPU_face|        %8.6f     %8.6f    (pure stagger error)\n",
        maximum(abs, stagger_err), maximum(abs, stagger_err[intslice]))
println()
println("  FaceInterpolateArray (FIA) uses the face-to-face index formula for dim 1,")
println("  eliminating the staggering offset.  Its interior error should match the")
println("  CenterField interior error (both are just linear interpolation truncation).")

# ══════════════════════════════════════════════════════════════════════════
# Part 2: Random source data (no analytic truth — compare IA vs FIA vs CPU)
# ══════════════════════════════════════════════════════════════════════════

using Random
Random.seed!(42)

rand_src = rand(Float32, Nλ_src, 1, 1)

rand_ia  = vec(Array(InterpolateArray(rand_src, tgt_sz, sharding,
                                      InterpolationType.Linear, (0, 0, 0))))
rand_fia = vec(Array(FaceInterpolateArray(rand_src, tgt_sz, sharding,
                                          InterpolationType.Linear, (0, 0, 0);
                                          face_dims=(true, false, false))))
rand_cpu = cpu_linear_periodic(vec(rand_src), Nλ_tgt, fi_face)

rand_ia_vs_cpu  = rand_ia  .- rand_cpu
rand_fia_vs_cpu = rand_fia .- rand_cpu

println()
println()
println("=" ^ 100)
println("RANDOM DATA: IA vs FIA vs CPU-periodic-face  (Nsrc=$Nλ_src → Ntgt=$Nλ_tgt)")
println("  (CPU periodic face-to-face is the reference — no analytic truth for random data)")
println("=" ^ 100)
@printf("  Max |IA − CPU_face|      all: %.6f   interior: %.6f   (stagger error)\n",
        maximum(abs, rand_ia_vs_cpu), maximum(abs, rand_ia_vs_cpu[intslice]))
@printf("  Max |FIA − CPU_face|     all: %.6f   interior: %.6f   (should be ≈ 0)\n",
        maximum(abs, rand_fia_vs_cpu), maximum(abs, rand_fia_vs_cpu[intslice]))
println()
@printf("  %3s  %10s  %10s  %10s  %10s  %10s\n",
        "i", "CPU_face", "IA", "IA_Δ", "FIA", "FIA_Δ")
@printf("  %3s  %10s  %10s  %10s  %10s  %10s\n",
        "---", "----------", "----------", "----------", "----------", "----------")
for i in 1:Nλ_tgt
    @printf("  %3d  %+10.6f  %+10.6f  %+10.6f  %+10.6f  %+10.6f\n",
            i, rand_cpu[i], rand_ia[i], rand_ia_vs_cpu[i],
            rand_fia[i], rand_fia_vs_cpu[i])
end

println()
@printf("  RANDOM SUMMARY (interior %d:%d):\n", first(intslice), last(intslice))
@printf("    IA  stagger error:  max %.6f   mean %.6f\n",
        maximum(abs, rand_ia_vs_cpu[intslice]),
        sum(abs, rand_ia_vs_cpu[intslice]) / length(intslice))
@printf("    FIA residual:       max %.6f   mean %.6f   ← FIX\n",
        maximum(abs, rand_fia_vs_cpu[intslice]),
        sum(abs, rand_fia_vs_cpu[intslice]) / length(intslice))
