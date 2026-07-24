#!/usr/bin/env bash
#
# Workflow script: export ncu report to CSV and analyze FLOPs.
#
# Reads the .ncu-rep from sharding/runs/ and outputs CSV + PNG into
# a subfolder of profile/ named after the run hash (e.g. 234_pJ4j).
#
# Requires the uenv to be loaded for access to ncu.
#
# Usage (from the profile/ directory):
#   # Option 1: If uenv is already loaded (ncu in PATH):
#   bash run_ncu_analysis.sh
#
#   # Option 2: Load uenv first:
#   uenv start --view=juliaup /capstor/scratch/cscs/lraess/uenv_julia/julia_26_3_v1_gh200.squashfs
#   bash run_ncu_analysis.sh
#
#   # Option 3: Analyze a different ncu report:
#   bash run_ncu_analysis.sh /path/to/ncu_profile.ncu-rep
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_DIR="${PROJECT_DIR}/sharding/runs"

# ── Configuration ─────────────────────────────────────────────────────
# Default: roofline run 2026-04-14T15-09-47.234_pJ4j
DEFAULT_RUN="2026-04-14T15-09-47.234_pJ4j"
RUN_NAME="${2:-${DEFAULT_RUN}}"
DEFAULT_NCU_REP="${RUNS_DIR}/${RUN_NAME}/ngpu=00001/ncu_profile.ncu-rep"
NCU_REP="${1:-${DEFAULT_NCU_REP}}"

# Derive output directory from run name: take everything after the date prefix
# e.g. "2026-04-14T15-09-47.234_pJ4j" -> "234_pJ4j"
RUN_HASH="${RUN_NAME##*.}"
OUT_DIR="${SCRIPT_DIR}/${RUN_HASH}"
mkdir -p "${OUT_DIR}"

# ncu binary — try PATH first, then the known uenv path
NCU_BIN="$(command -v ncu 2>/dev/null || true)"
if [[ -z "${NCU_BIN}" ]]; then
    NCU_BIN="/user-environment/linux-neoverse_v2/cuda-12.9.1-3mm7luhvmoio6jtxxxj4mnayoe64pgr7/bin/ncu"
fi

# Output CSV in the output directory (filename includes run hash)
CSV_OUT="${OUT_DIR}/ncu_profile_raw_${RUN_HASH}.csv"

# ── Validate ──────────────────────────────────────────────────────────
if [[ ! -f "${NCU_REP}" ]]; then
    echo "ERROR: ncu report not found: ${NCU_REP}" >&2
    exit 1
fi

if [[ ! -x "${NCU_BIN}" ]]; then
    echo "ERROR: ncu binary not found. Load the uenv first:" >&2
    echo "  uenv start --view=juliaup /capstor/scratch/cscs/lraess/uenv_julia/julia_26_3_v1_gh200.squashfs" >&2
    exit 1
fi

echo "ncu binary:  ${NCU_BIN}"
echo "ncu report:  ${NCU_REP}"
echo "Output dir:  ${OUT_DIR}"
echo "CSV output:  ${CSV_OUT}"
echo

# ── Step 1: Export ncu report to CSV ──────────────────────────────────
echo "── Step 1: Exporting ncu report to CSV..."
"${NCU_BIN}" -i "${NCU_REP}" --csv --page raw 2>/dev/null > "${CSV_OUT}"
NLINES=$(wc -l < "${CSV_OUT}")
echo "   Exported ${NLINES} lines (header + units + $((NLINES - 2)) kernel launches)"
echo

# ── Step 2: Run FLOPs analysis ────────────────────────────────────────
echo "── Step 2: Analyzing FLOPs..."
echo

# Use julia11 if available (ALPS alias), otherwise julia
JULIA_BIN="$(command -v julia11 2>/dev/null || command -v julia 2>/dev/null || true)"
if [[ -z "${JULIA_BIN}" ]]; then
    echo "ERROR: julia not found in PATH" >&2
    exit 1
fi

"${JULIA_BIN}" --project="${SCRIPT_DIR}" "${SCRIPT_DIR}/analyze_ncu_flops.jl" "${CSV_OUT}" --top 20

# ── Step 3: Generate roofline plot ────────────────────────────────────
echo
echo "── Step 3: Generating roofline plot (PNG)..."
"${JULIA_BIN}" --project="${SCRIPT_DIR}" "${SCRIPT_DIR}/plot_roofline.jl" "${CSV_OUT}" --output "${OUT_DIR}/roofline_${RUN_HASH}.png" --top 20
