include("common_submission_generator.jl")

# Perlmutter launcher for correctness_interpolation_atmosphere_simulation_run.jl.
# Submits a single Ngpu=4 job (Rx=Ry=2 is hardcoded in the correctness script)
# that compares the Reactant/sharded IC loader against the vanilla CPU IC loader.

account = "m5176"

queue = "regular"
out_dir = joinpath(ENV["SCRATCH"], "GB25")

submit   = true
run_name = "r_corr_"
time     = "01:00:00"

# Correctness script is sharding-aware but hardcodes Rx=Ry=2, so only Ngpu=4 works.
Ngpus         = [4]
type          = "weak"
gpus_per_node = 4
cpus_per_task = 16

perlmutter_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                              run_name, gpus_per_node, type, submit)

function perlmutter_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                      resolution_fraction, project_path, run_file)

    # Smaller grid than the production scaling test — correctness only needs
    # enough resolution to exercise the interpolation math, not peak memory.
    # Nλ = grid-x * Rx - 2H = 128*2 - 8 = 248, same for Nφ.
    x, y, z = (128, 128, 64)

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

source /global/common/software/nersc9/julia/scripts/activate_beta.sh
ml load julia/1.11.7

module load nccl/2.29.2-cu13

export SBATCH_ACCOUNT=$(cfg.account)
export SALLOC_ACCOUNT=$(cfg.account)
export JULIA_CUDA_MEMORY_POOL=none

# HACKS to get this to work on Perlmutter
export LD_PRELOAD=/usr/lib64/libstdc++.so.6

export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
export NCCL_BUFFSIZE=33554432
export JULIA_CUDA_USE_COMPAT=false

srun -n $(Nnodes) -c 32 -G $(Ngpu) --cpu-bind=verbose,cores \
    $(job_dir)/launcher.sh \
    $(Base.julia_cmd()[1]) --project=$(project_path) --compiled-modules=strict -O0 \
    $(run_file) --grid-x $(x) --grid-y $(y) --grid-z $(z)
"""
end

generate_and_submit(perlmutter_submit_job_writer, perlmutter_config; caller_file=@__FILE__)
