#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export REACTANT_VISIBLE_GPU_DEVICES="${REACTANT_VISIBLE_GPU_DEVICES:-0,1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=host-offload-legalize,hlo_constant_splitter,multi_output_fusion --xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80}"
export XLA_REACTANT_GPU_MEM_FRACTION="${XLA_REACTANT_GPU_MEM_FRACTION:-0.20}"
export XLA_REACTANT_GPU_PREALLOCATE="${XLA_REACTANT_GPU_PREALLOCATE:-false}"

exec julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_instability_simulation_run.jl \
  --float-type=Float64 \
  --target-float-type=Float32 \
  --dimension=first \
  "$@"
