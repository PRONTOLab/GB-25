#!/bin/bash
# Parse a correctness_interpolation_atmosphere_simulation_run.jl .out file
# and report per-stage pass/fail with the worst max|δ| per stage.
#
# Usage: parse_correctness.sh <path/to/job.out>

set -euo pipefail

OUT="${1:?Usage: $0 <out file>}"
test -f "$OUT" || { echo "no such file: $OUT"; exit 1; }

# Stage headers we expect, in order:
#   "After file-based IC loading"
#   "After first time step"
#   "After 10 steps"  (script uses Nt = 10)
#   "After ... steps" or "After a loop"

awk '
function flush_stage(   name, fail_count) {
    if (stage == "") return
    fail_count = 0
    for (k in stage_fail) if (stage_fail[k]) fail_count++
    printf "── stage: %s ──\n", stage
    printf "  fields compared: %d\n", stage_count
    printf "  fields failing : %d\n", fail_count
    printf "  worst max|δ|   : %.6e (field %s)\n", stage_worst, stage_worst_field
    if (stage_summary != "") printf "  summary line   : %s\n", stage_summary
    print ""
    stage = ""; stage_count = 0; stage_worst = 0; stage_worst_field = ""; stage_summary = ""
    delete stage_fail
}

/^\[ Info: After/ {
    flush_stage()
    stage = $0
    sub(/^\[ Info: /, "", stage)
    next
}

# Match lines like "(   ρ) ψ₁ ≈ ψ₂: true , ..., max|δ|: 1.234e-08 at 1 2 3"
# or false. Extract field name, pass/fail, and max|δ|.
/ψ₁ ≈ ψ₂/ {
    fld = $0
    sub(/^\(/, "", fld); sub(/\).*/, "", fld); gsub(/ /, "", fld)
    pf = ($0 ~ /ψ₁ ≈ ψ₂: true/) ? "true" : "false"
    # max|δ|: <num>
    md = $0; sub(/^.*max\|δ\|: */, "", md); sub(/ .*/, "", md)
    md_num = md + 0.0
    stage_count++
    stage_fail[fld] = (pf == "false") ? 1 : 0
    if (md_num > stage_worst) { stage_worst = md_num; stage_worst_field = fld }
    next
}

/The two atmosphere models are consistent/ { stage_summary = "PASS — consistent"; next }
/discrepancy between the atmosphere models/ { stage_summary = "FAIL — discrepancy"; next }

END { flush_stage() }
' "$OUT"

echo "── overall ──"
PASS=$(grep -c "The two atmosphere models are consistent" "$OUT" || true)
FAIL=$(grep -c "discrepancy between the atmosphere models" "$OUT" || true)
echo "  passed stages : $PASS"
echo "  failed stages : $FAIL"

# Surface fatal errors / NaN traces if any
echo "── errors / NaNs in log ──"
grep -nE "ERROR|NaN|Stacktrace|Out of memory|CUDA error|XLA error" "$OUT" | head -20 || echo "  (none)"
