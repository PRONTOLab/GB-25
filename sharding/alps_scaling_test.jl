include("common_submission_generator.jl")

out_dir = @__DIR__

# julia "module" on ALPS
# uenv start --view=juliaup,modules julia/25.5:v1
# may be needed on ALPS with Julia 1.11
# export LD_PRELOAD=/capstor/scratch/cscs/lraess/.julia/gh200/juliaup/depot/artifacts/152ab7c1cf7e3e69c2fa76110b9e01affcbb1f36/lib/libcrypto.so.3

# run params
account  = "c44"
submit   = true #false
run_name = "reactant_"
time     = "00:30:00"
Ngpus    = [4, 8, 12, 16]
type     = "weak"

gpus_per_node = 4
cpus_per_task = 288

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
# #SBATCH --reservation=$(account)
#SBATCH --exclusive

export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_USE_COMPAT=false
export FI_MR_CACHE_MONITOR=disabled

ulimit -s unlimited

# Setting `--cpu_bind` is explicitly discouraged:
# <https://eth-cscs.github.io/cscs-docs/guides/gb2025/#slurm>.
# We only set it to `verbose` to record what's going on.
srun $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) --startup-file=no --threads=16 --compiled-modules=strict -O0 $(run_file)
"""
end

generate_and_submit(alps_submit_job_writer, alps_config; caller_file=@__FILE__)
