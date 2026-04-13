include("common_submission_generator.jl")

out_dir = @__DIR__

# julia "module" on ALPS
# uenv start --view=juliaup,modules julia/25.5:v1
# may be needed on ALPS with Julia 1.11
# export LD_PRELOAD=/capstor/scratch/cscs/lraess/.julia/gh200/juliaup/depot/artifacts/152ab7c1cf7e3e69c2fa76110b9e01affcbb1f36/lib/libcrypto.so.3

# run params
account  = "g209"
submit   = true #false
run_name = "reactant_"
time     = "00:40:00"

# We want to preserve a 2:1 aspect ratio for the x:y dimensions in all runs
# so we pick Ngpu from the set of numbers 8*n^2 where n is any integer.
# We also try to pick the those numbers which are as close as possible to powers of 2,
# and such that the sum of all the numbers is less than 2*8192 (so they can be run simultaneously).
# Also 9180 is chosen specifically because it is the alps system size
# Ngpus     = [4, 8, 16, 32, 72, 128, 288, 512, 968, 2048, 3872, 8192, 9152]
Ngpus = [4, 8, 16, 32, 72, 128]

type     = "weak"

gpus_per_node = 4
cpus_per_task = 288

alps_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                        run_name, gpus_per_node, type, submit)

function alps_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                resolution_fraction, project_path, run_file)

    x, y = (768, 768)

    """
#!/bin/bash -l

#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(time)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
#SBATCH --nodes=$(Nnodes)
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=$(gpus_per_node)
#SBATCH --gpu-bind=per_task:$(gpus_per_node)
#SBATCH --constraint=gpu
#SBATCH --account=$(account)
# #SBATCH --reservation=$(account)
#SBATCH --exclusive

export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_USE_COMPAT=false
export FI_MR_CACHE_MONITOR=disabled

# https://docs.cscs.ch/software/communication/nccl/#uenv
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
# export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NCHANNELS_PER_NET_PEER=4

# Tell NCLL to use fast network, this may solve some rendezvous failures
export NCCL_SOCKET_IFNAME="hsn"
    
# Equivalent to loading the `aws-ofi-nccl` module, without having to load it:
# https://docs.cscs.ch/software/communication/nccl/#uenv
export LD_LIBRARY_PATH="/user-environment/linux-neoverse_v2/aws-ofi-nccl-1.17.1-rpvjytyqpdw2taig4xibhrtgudie4a3q/lib:/user-environment/linux-neoverse_v2/libfabric-2.3.1-npwd54pnpalgjcizhpejkh7gwg4c7idu/lib:/user-environment/linux-neoverse_v2/aws-ofi-nccl-1.17.1-rpvjytyqpdw2taig4xibhrtgudie4a3q/lib"

ulimit -s unlimited

# Disable core dumps: https://docs.cscs.ch/guides/gb2026/#disabling-core-dumps
ulimit -S -c0

# Setting `--cpu_bind` is explicitly discouraged:
# <https://eth-cscs.github.io/cscs-docs/guides/gb2025/#slurm>.
# We only set it to `verbose` to record what's going on.
srun --uenv="\${SCRATCH}/uenv_julia/julia_26_3_v1_gh200.squashfs" --view=juliaup --preserve-env --cpu_bind=verbose \
    --export=ALL,LD_PRELOAD="/user-environment/linux-neoverse_v2/nccl-2.28.7-1-sybuzb6n6j63b2pazvl2vh3nktz3jq27/lib/libnccl.so.2" \
    $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) --startup-file=no --threads=16 --compiled-modules=strict -O0 $(run_file) --grid-x $(x) --grid-y $(y) --grid-z 64
"""
end

generate_and_submit(alps_submit_job_writer, alps_config; caller_file=@__FILE__)
