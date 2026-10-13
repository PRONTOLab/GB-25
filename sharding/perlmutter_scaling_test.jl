include("common_submission_generator.jl")

account = "m4672"
account = "m5096"
account = "m5176"

queue = "regular"
out_dir = joinpath(ENV["SCRATCH"], "GB25")

# run params
submit   = true
run_name = "r_react_"
time     = "00:40:00"

# We want to preserve a 2:1 aspect ratio for the x:y dimensions in all runs
# so we pick Ngpus from the set of numbers 8*n^2 where n is any integer.
# We also try to pick the those numbers which are as close as possible to powers of 2,
# and such that the sum of all the numbers is less than 2*8192 (so they can be run simultaneously).
# Ngpus     = [4, 8, 32, 72, 128, 288, 512, 968, 2048, 3872, 6136]
Ngpus     = [4, 8, 32, 72, 128, 288, 512, 968, 2048, 6136]
Ngpus     = [6136]
Ngpus     = [4]
Ngpus     = [4, 8, 32, 72, 128]
Ngpus     = [2, 4, 8]
comms_opts = ["off", "on"]

type     = "weak"

gpus_per_node = 4
cpus_per_task = 16

perlmutter_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus, comms_opts,
                              run_name, gpus_per_node, type, submit)

function perlmutter_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                      resolution_fraction, project_path, run_file,
                                      comm::String)

    # # grid sizes for sharded_baroclinic_instability_simulation_run.jl
    # x, y = (256,256) # fits easily
    # # x, y = (320, 320) # should fit fine, peak in use 21GB on 512 GPU
    # # x, y = (384, 384) # seems to run fine in most cases, but might be close, has immediate returned before, peak in use nearly 30GB on 288GPU

    # # grid sizes for sharded_atmosphere_simulation_run.jl
    # x, y, z = (576, 576, 64) # might be better off doing something like this, should be about 23GB
    # x, y, z = (640, 640, 64) # gives Peak In Use: 30.446 GiB might be risky
    # x, y, z = (672, 672, 64) # allocates about 30GB/GPU, probably don't want to go much higher
    x, y, z = (512, 512, 64) # per-GPU size for 1°-IC Reactant test

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
#SBATCH --mail-user=email@solidcompany.com
#SBATCH --mail-type=ALL

module reset
source /global/common/software/nersc9/julia/scripts/activate_beta.sh
ml load julia/1.11.7
module load nccl/2.29.2-cu13
module list

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

export GB25_COMMS_OPT=$(comm)

if [ "\${PROFILER:-}" = "nsys" ]; then
    OPENSSL_LIB="\${GB25_OPENSSL_LIB:-\$(find "\${PERLMUTTER_DEPOT:-\${SCRATCH:-}/GB25/perlmutter-2026-07-16-depot}/artifacts" -maxdepth 3 -name 'libcrypto.so.3' -print -quit 2>/dev/null)}"
    PROFILER_CMD="nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --output=$(job_dir)/nsys-job-%q{SLURM_JOB_ID}-rank-%q{SLURM_PROCID} env GB25_NSYS=true JULIA_CUDA_NSYS=nsys \${OPENSSL_LIB:+LD_PRELOAD=\${OPENSSL_LIB}:\${LD_PRELOAD}}"
elif [ "\${PROFILER:-}" = "xprof" ]; then
    export GB25_XPROF=true
fi

srun -n $(Nnodes) -c 32 -G $(Ngpu) --cpu-bind=verbose,cores \\
    \${PROFILER_CMD} $(job_dir)/launcher.sh \\
    $(Base.julia_cmd()[1]) --project=$(project_path) --compiled-modules=strict -O0 \\
    $(run_file) --grid-x $(x) --grid-y $(y) --grid-z $(z)
"""
end

generate_and_submit(perlmutter_submit_job_writer, perlmutter_config; caller_file=@__FILE__)
