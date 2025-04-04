include("common_submission_generator.jl")

out_dir = @__DIR__

# run params
account  = "c44" #"g191"
submit   = true
run_name = "reactant_"
time     = "01:00:00"
Ngpus    = [64, 256] #[4, 8, 12, 16]
type     = "weak"

gpus_per_node = 4
cpus_per_task = 16

alps_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                        run_name, gpus_per_node, type, submit)

function alps_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                resolution_fraction, project_path, run_file)

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

export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_USE_COMPAT=false
export FI_MR_CACHE_MONITOR=disabled

ulimit -s unlimited

# Setting `--cpu_bind` is explicitly discouraged:
# <https://eth-cscs.github.io/cscs-docs/guides/gb2025/#slurm>.
# We only set it to `verbose` to record what's going on.
# srun --uenv=prgenv-gnu --preserve-env --cpu_bind=verbose \
#     --export=ALL,LD_PRELOAD="/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd/lib/libnccl.so.2" \
#     $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) --startup-file=no --threads=$(cfg.cpus_per_task) -O0 $(run_file)

srun --preserve-env --cpu_bind=verbose \
    --export=ALL,LD_PRELOAD="/user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nccl-2.20.3-1-kogp7m7qjex2wykzdnudvs6x3r4dchpn//lib/libnccl.so.2" \
    $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) --startup-file=no --threads=$(cfg.cpus_per_task) -O0 $(run_file)
"""
end

generate_and_submit(alps_submit_job_writer, alps_config; caller_file=@__FILE__)
