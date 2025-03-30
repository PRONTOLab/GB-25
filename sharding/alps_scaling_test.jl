include("common_submission_generator.jl")

out_dir = @__DIR__

# run params
account  = "g191"
submit   = false
run_name = "reactant_"
time     = "00:40:00"
Ngpus    = [4, 8] #[4, 8, 12, 16]
type     = "weak"

gpus_per_node = 4
cpus_per_task = 16

alps_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                        run_name, gpus_per_node, type, submit)

function alps_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                resolution_fraction, project_path, run_file)

    MPICH_GPU_SUPPORT_ENABLED = 1

    """
#!/bin/bash -l

#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(time)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
#SBATCH --nodes=$(Nnodes)
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=$gpus_per_node
#SBATCH --constraint=gpu
#SBATCH --account=$(account)
#SBATCH --exclusive


export Ngpu=$(Ngpu)
export resolution_fraction=$(resolution_fraction)
export JULIA_DEBUG="Reactant,Reactant_jll"
export JULIA_DEPOT_PATH=$(join(Base.DEPOT_PATH, ':'))
# export TF_CPP_MAX_VLOG_LEVEL=3
# export XLA_FLAGS="--xla_dump_to=$(job_dir)/xla_dump"
export MPICH_GPU_SUPPORT_ENABLED=$(MPICH_GPU_SUPPORT_ENABLED)

ulimit -s unlimited

# Setting `--cpu_bind` is explicitly discouraged:
# <https://eth-cscs.github.io/cscs-docs/guides/gb2025/#slurm>.
# We only set it to vrbose to record what's going on
srun --uenv=prgenv-gnu --view=prgenv-gnu:default --preserve-env --gpu-bind=per_task:1 --cpu_bind=verbose \
    --export=ALL,LD_PRELOAD="/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd/lib/libnccl.so.2" \
    $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) --startup-file=no --threads=$(cfg.cpus_per_task) -O0 $(run_file)
"""
end

generate_and_submit(alps_submit_job_writer, alps_config; caller_file=@__FILE__)
