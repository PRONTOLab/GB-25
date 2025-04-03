include("common_submission_generator.jl")

account = "m4672"
out_dir = joinpath(ENV["SCRATCH"], "GB25")

# run params
submit   = true
run_name = "r_react_"
time     = "01:00:00"
Ngpus    = [4, 8, 16, 32, 64, 128]
type     = "weak"

gpus_per_node = 4
cpus_per_task = 16

perlmutter_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                              run_name, gpus_per_node, type, submit)

function perlmutter_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                      resolution_fraction, project_path, run_file)

                """
#!/bin/bash -l

#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpu-bind=none
#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(cfg.time)
#SBATCH --nodes=$(Nnodes)
#SBATCH --account=$(cfg.account)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err

source /global/common/software/nersc9/julia/scripts/activate_beta.sh
ml load julia/1.10.9

export SBATCH_ACCOUNT=$(cfg.account)
export SALLOC_ACCOUNT=$(cfg.account)
export JULIA_CUDA_MEMORY_POOL=none

srun -n $(Nnodes) -c 32 -G $(Ngpu) --cpu-bind=verbose,cores $(job_dir)/launcher.sh julia --project=$(project_path) -O0 $(run_file) 
"""
end

generate_and_submit(perlmutter_submit_job_writer, perlmutter_config; caller_file=@__FILE__)
