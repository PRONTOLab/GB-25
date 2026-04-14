#!/bin/bash -l
#SBATCH --job-name="GB25_surface"
#SBATCH --time=00:30:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=18
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
export LD_LIBRARY_PATH="/user-environment/linux-neoverse_v2/aws-ofi-nccl-1.17.1-rpvjytyqpdw2taig4xibhrtgudie4a3q/lib:/user-environment/linux-neoverse_v2/libfabric-2.3.1-npwd54pnpalgjcizhpejkh7gwg4c7idu/lib"
ulimit -s unlimited
ulimit -S -c0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TZ=UTC
export XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=600 --xla_gpu_first_collective_call_terminate_timeout_seconds=600"
export XLA_FLAGS="--xla_disable_hlo_passes=host-offload-legalize,hlo_constant_splitter,multi_output_fusion ${XLA_FLAGS}"
export XLA_REACTANT_GPU_MEM_FRACTION=0.9
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
srun --uenv="${SCRATCH}/uenv_julia/julia_26_3_v1_gh200.squashfs" \
     --view=juliaup --preserve-env --cpu_bind=verbose \
     --export=ALL,LD_PRELOAD="/user-environment/linux-neoverse_v2/nccl-2.28.7-1-sybuzb6n6j63b2pazvl2vh3nktz3jq27/lib/libnccl.so.2" \
     "${PROJECT_DIR}/sharding/julia_wrapper.sh" --project="${PROJECT_DIR}" --startup-file=no --threads=16 -O0 \
     "${PROJECT_DIR}/sharding/test_one_step_save.jl" --grid-x 512 --grid-y 512 --grid-z 64
