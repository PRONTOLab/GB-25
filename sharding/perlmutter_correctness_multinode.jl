include("common_submission_generator.jl")

# Multi-node Reactant-vs-vanilla-CPU correctness comparison.
#
# Reuses sharding/correctness_interpolation_atmosphere_simulation_run.jl which
# now picks the partition from factors(Ndev). Per-device size is intentionally
# small so the CPU vanilla model still fits within walltime at the matching
# total grid:
#
#   Ngpu=4  → factors=(2,2)  → total = (4*2, 4*2)*64  = 256x256
#   Ngpu=8  → factors=(4,2)  → total = (4*4, 4*2)*64  = 256x128
#   Ngpu=16 → factors=(4,4)  → total = (4*4, 4*4)*64  = 256x256
#
# The 64x64 per-device choice keeps total cells modest so vanilla CPU runs
# the comparison stages in well under the walltime.

account = "m5176"
queue   = "regular"
out_dir = joinpath(ENV["SCRATCH"], "GB25")

submit   = true
run_name = "r_corrmn_"
time     = "01:00:00"

# Multi-node sweep. 4 is included as a sanity check that the generalized
# partition still matches the original single-node correctness behavior.
Ngpus         = [8, 16]
type          = "weak"
gpus_per_node = 4
cpus_per_task = 16

perlmutter_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                              run_name, gpus_per_node, type, submit)

function perlmutter_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                      resolution_fraction, project_path, run_file)

    # Per-device grid (small so the matching CPU vanilla model is tractable).
    x, y, z = (64, 64, 64)

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
