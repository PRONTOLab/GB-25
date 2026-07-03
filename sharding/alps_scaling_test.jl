include("common_submission_generator.jl")

out_dir = @__DIR__

# Current config requires `uenv start julia/26.3:v1 --view=juliaup,modules` to be loaded before running the
# submission script: julia +1.11 --project alps_scaling_test.jl sharded_baroclinic_instability_simulation_run.jl

# run params
account  = "c44"
submit   = true #false
run_name = "reactant_"
time     = "00:30:00"

# We want to preserve a 2:1 aspect ratio for the x:y dimensions in all runs
# so we pick Ngpu from the set of numbers 8*n^2 where n is any integer.
# We also try to pick the those numbers which are as close as possible to powers of 2,
# and such that the sum of all the numbers is less than 2*8192 (so they can be run simultaneously).
# Also 9180 is chosen specifically because it is the alps system size
Ngpus     = [4, 8]
# Ngpus     = [4, 8, 16, 32, 72, 128, 288, 512, 968, 2048, 3872, 8192, 9152]

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
#SBATCH --exclusive
#SBATCH --uenv-passthrough=use

export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_USE_COMPAT=false

# https://docs.cscs.ch/software/communication/nccl/#uenv
export NCCL_NCHANNELS_PER_NET_PEER=4

# Tell NCLL to use fast network, this may solve some rendezvous failures
export NCCL_SOCKET_IFNAME="hsn"

ulimit -s unlimited
# Disable core dumps: https://docs.cscs.ch/guides/gb2026/#disabling-core-dumps
ulimit -S -c0

# We only set it to `verbose` to record what's going on.
srun --preserve-env --cpu_bind=verbose \
    $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) --startup-file=no --threads=16 --compiled-modules=strict -O0 $(run_file) --grid-x $(x) --grid-y $(y) --grid-z 64
"""
end

generate_and_submit(alps_submit_job_writer, alps_config; caller_file=@__FILE__)
