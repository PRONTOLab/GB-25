include("common_submission_generator.jl")

account = "m4672"
account = "m5096"
account = "m5176"

queue = "regular"
out_dir = joinpath(ENV["SCRATCH"], "GB25")

# run params
submit   = true
run_name = "r_react_"
time     = "01:00:00"

# We want to preserve a 2:1 aspect ratio for the x:y dimensions in all runs
# so we pick Ngpus from the set of numbers 8*n^2 where n is any integer.
# We also try to pick the those numbers which are as close as possible to powers of 2,
# and such that the sum of all the numbers is less than 2*8192 (so they can be run simultaneously).
Ngpus     = [4, 8, 32, 72, 128, 288, 512, 968, 2048, 3872, 8192]
Ngpus     = [4]

type     = "weak"

gpus_per_node = 4
cpus_per_task = 16

perlmutter_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                              run_name, gpus_per_node, type, submit)

function perlmutter_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                      resolution_fraction, project_path, run_file)

    # # grid sizes for sharded_baroclinic_instability_simulation_run.jl
    # x, y, z = (256,256, 128) # the largest grid that I know fits
    # # x, y, z = (320, 320, 128) # think might also fit, but the test crashed for other reasons
    # # x, y, z = (384, 384, 128) # pretty sure this doesn't quite fit

    # # grid sizes for sharded_atmosphere_simulation_run.jl
    # x, y, z = (568, 568, 64)
    x, y, z = (640, 640, 64) # gives Peak In Use: 30.446 GiB
    # x, y, z = (672, 672, 64) # allocates about 30GB/GPU, probably don't want to go much higher

#SBATCH -q premium
                """
#!/bin/bash -l

#SBATCH -C gpu&hbm40g
#SBATCH -q $(queue)
#SBATCH --gpu-bind=none
#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(cfg.time)
#SBATCH --nodes=$(Nnodes)
#SBATCH --account=$(cfg.account)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
# #SBATCH --mail-user=email@email.gov
# #SBATCH --mail-type=ALL

source /global/common/software/nersc9/julia/scripts/activate_beta.sh
ml load julia/1.11.7

module load nccl/2.29.2-cu13

export SBATCH_ACCOUNT=$(cfg.account)
export SALLOC_ACCOUNT=$(cfg.account)
export JULIA_CUDA_MEMORY_POOL=none

#
# HACKS to get this to work on Perlmutter
#

# fixes incompatability btwn system julia and nccl
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
# export MPICH_SMP_SINGLE_COPY_MODE=NONE
# export NCCL_DEBUG=INFO
# export FI_MR_CACHE_MONITOR=kdreg2
# export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_BUFFSIZE=33554432
export JULIA_CUDA_USE_COMPAT=false

srun -n $(Nnodes) -c 32 -G $(Ngpu) --cpu-bind=verbose,cores \
    $(job_dir)/launcher.sh \
    $(Base.julia_cmd()[1]) --project=$(project_path) --compiled-modules=strict -O0 \
    $(run_file) --grid-x $(x) --grid-y $(y) --grid-z $(z)
"""
end

generate_and_submit(perlmutter_submit_job_writer, perlmutter_config; caller_file=@__FILE__)
