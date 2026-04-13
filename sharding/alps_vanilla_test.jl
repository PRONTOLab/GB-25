include("common_submission_generator.jl")

out_dir = @__DIR__

account  = "g209"
submit   = true
run_name = "vanilla_"
time     = "00:30:00"

Ngpus     = [4]

type     = "weak"

gpus_per_node = 4
cpus_per_task = 72   # 288 / 4 tasks per node

alps_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                        run_name, gpus_per_node, type, submit)

function alps_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                resolution_fraction, project_path, run_file)

    x, y = (768, 768)
    tasks_per_node = min(Ngpu, gpus_per_node)
    depot_path = join(Base.DEPOT_PATH, ':')

    """
#!/bin/bash -l

#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(time)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
#SBATCH --nodes=$(Nnodes)
#SBATCH --ntasks-per-node=$(tasks_per_node)
#SBATCH --gpus-per-node=$(gpus_per_node)
#SBATCH --gpu-bind=per_task:1
#SBATCH --constraint=gpu
#SBATCH --account=$(account)
#SBATCH --exclusive

export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_USE_COMPAT=false
export FI_MR_CACHE_MONITOR=disabled
export TZ=UTC
export JULIA_DEPOT_PATH=$(depot_path)

ulimit -s unlimited
ulimit -S -c0

srun --mpi=pmi2 --uenv="\${SCRATCH}/uenv_julia/julia_26_3_v1_gh200.squashfs" --view=juliaup --preserve-env --cpu_bind=verbose \
    $(Base.julia_cmd()[1]) --project=$(project_path) --startup-file=no --threads=16 --compiled-modules=strict -O0 $(run_file) --grid-x $(x) --grid-y $(y) --grid-z 64
"""
end

generate_and_submit(alps_submit_job_writer, alps_config; caller_file=@__FILE__)
