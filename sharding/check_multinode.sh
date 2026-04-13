#!/bin/bash -l

#SBATCH --job-name="check_interp"
#SBATCH --time=00:10:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=per_task:4
#SBATCH --constraint=gpu
#SBATCH --account=g209
#SBATCH --exclusive

export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_USE_COMPAT=false
export FI_MR_CACHE_MONITOR=disabled

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
export LD_LIBRARY_PATH="/user-environment/linux-neoverse_v2/aws-ofi-nccl-1.17.1-rpvjytyqpdw2taig4xibhrtgudie4a3q/lib:/user-environment/linux-neoverse_v2/libfabric-2.3.1-npwd54pnpalgjcizhpejkh7gwg4c7idu/lib:/user-environment/linux-neoverse_v2/aws-ofi-nccl-1.17.1-rpvjytyqpdw2taig4xibhrtgudie4a3q/lib"

ulimit -s unlimited
ulimit -S -c0

PROJECT_DIR="/iopsstor/scratch/cscs/dkytezab/GB-25-atmos"
CHECK_SCRIPT="${PROJECT_DIR}/sharding/check.jl"

srun --uenv="${SCRATCH}/uenv_julia/julia_26_3_v1_gh200.squashfs" --view=juliaup --preserve-env --cpu_bind=verbose \
    --export=ALL,LD_PRELOAD="/user-environment/linux-neoverse_v2/nccl-2.28.7-1-sybuzb6n6j63b2pazvl2vh3nktz3jq27/lib/libnccl.so.2" \
    julia --project="$PROJECT_DIR" --startup-file=no --threads=16 -O0 "$CHECK_SCRIPT" --grid-x 64 --grid-y 64 --grid-z 8
