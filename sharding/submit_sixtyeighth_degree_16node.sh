#!/bin/bash -l
#
# 1/68° NCCL-distributed atmosphere run — 64 GPUs across 16 nodes on CSCS Alps.
# Grid: Nλ=24480, Nφ=10880, Nz=64 → 17 billion cells, 266M cells/rank
# Partition: Rx=8, Ry=8, Δt=0.2s, 54000 iters (3 sim hours)
# Estimated wall time: ~7.5 hours (+1h buffer)
#

#SBATCH --job-name="GB25_68th_deg_64gpu"
#SBATCH --time=09:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --account=g209
#SBATCH --exclusive

# ── MPI / libfabric ──────────────────────────────────────────────────────
export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_USE_COMPAT=false
export FI_MR_CACHE_MONITOR=disabled

# ── NCCL tuning for Alps (AWS Libfabric over Slingshot) ──────────────────
# https://docs.cscs.ch/software/communication/nccl/#uenv
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_SOCKET_IFNAME="hsn"

# AWS OFI NCCL library paths
export LD_LIBRARY_PATH="/user-environment/linux-neoverse_v2/aws-ofi-nccl-1.17.1-rpvjytyqpdw2taig4xibhrtgudie4a3q/lib:/user-environment/linux-neoverse_v2/libfabric-2.3.1-npwd54pnpalgjcizhpejkh7gwg4c7idu/lib:/user-environment/linux-neoverse_v2/aws-ofi-nccl-1.17.1-rpvjytyqpdw2taig4xibhrtgudie4a3q/lib"

ulimit -s unlimited
ulimit -S -c0

# ── Environment ──────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TZ=UTC

# Disable proxy env vars that can cause XLA/NCCL hangs
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

# ── Launch ───────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_SCRIPT="${PROJECT_DIR}/sharding/sixtyeighth_degree_nccl_distributed_run.jl"

srun --uenv="${SCRATCH}/uenv_julia/julia_26_3_v1_gh200.squashfs" \
     --view=juliaup \
     --preserve-env \
     --cpu_bind=verbose \
     --export=ALL,LD_PRELOAD="/user-environment/linux-neoverse_v2/nccl-2.28.7-1-sybuzb6n6j63b2pazvl2vh3nktz3jq27/lib/libnccl.so.2" \
     julia --project="${PROJECT_DIR}" --startup-file=no -O0 "${RUN_SCRIPT}"
