include("common_submission_generator.jl")

account = "try25_giordano"
out_dir = @__DIR__

# run params
submit   = false
run_name = "reactant_"
time     = "00:40:00"
Ngpus    = [4, 8, 16, 32]
type     = "weak"

gpus_per_node = 4
cpus_per_task = 16

leonardo_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                            run_name, gpus_per_node, type, submit)

function leonardo_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                    resolution_fraction, project_path, run_file)

    """
#!/bin/bash -l

#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(cfg.time)
#SBATCH --nodes=$(Nnodes)
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:$(gpus_per_node)
#SBATCH --cpus-per-task=$(cfg.cpus_per_task)
#SBATCH --partition boost_usr_prod
#SBATCH --account=$(cfg.account)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
#SBATCH --qos=$(Nnodes <= 64 ? "normal" : "boost_qos_bprod")
#SBATCH --gpu-bind=none

# Make ultra sure this env var isn't set before loading CUDA module
export LD_LIBRARY_PATH=""
unset LD_LIBRARY_PATH
# export SBATCH_ACCOUNT=$(cfg.account)
# export SALLOC_ACCOUNT=$(cfg.account)
export JULIA_CUDA_MEMORY_POOL=none
export Ngpu=$(Ngpu)
export resolution_fraction=$(resolution_fraction)
export JULIA_DEBUG="Reactant,Reactant_jll"
export JULIA_DEPOT_PATH=$(join(Base.DEPOT_PATH, ':'))
# export TF_CPP_MAX_VLOG_LEVEL=3
# export XLA_FLAGS="--xla_dump_to=$(job_dir)/xla_dump"

module load cuda/12.3
srun --cpu-bind=verbose,cores $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) -O0 $(run_file)
"""
end

generate_and_submit(leonardo_submit_job_writer, leonardo_config; caller_file=@__FILE__)
