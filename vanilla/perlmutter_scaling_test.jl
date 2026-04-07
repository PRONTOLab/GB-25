include("common_submission_generator.jl")

account = "m4672"
account = "m5096"
account = "m5176"

queue = "regular"
queue = "debug"
out_dir = joinpath(ENV["SCRATCH"], "GB25")

# run params
submit   = true
run_name = "r_react_"
time     = "00:15:00"
Ngpus    = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
Ngpus    = [4]
Ngpus    = [4, 8]
type     = "weak"

gpus_per_node = 4
cpus_per_task = 16

perlmutter_config = JobConfig(; username, account, out_dir, time, cpus_per_task, Ngpus,
                              run_name, gpus_per_node, type, submit)

function perlmutter_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                      resolution_fraction, project_path, run_file)

#SBATCH -q premium
                """
#!/bin/bash -l

#SBATCH -C gpu&hbm40g
#SBATCH -q $(queue)
#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(cfg.time)
#SBATCH --nodes=$(Nnodes)
#SBATCH --account=$(cfg.account)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err

module purge
module load PrgEnv-gnu/8.6.0
module load gpu
module load julia/1.11.7

# OpenMP settings:
# probably not necessary but can't hurt
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# configure cuda-aware mpi x julia x cray mpich
export JULIA_NUM_THREADS=1
export JULIA_CUDA_MEMORY_POOL=none
export MPICH_GPU_SUPPORT_ENABLED=1

export SBATCH_ACCOUNT=m5176
export SALLOC_ACCOUNT=m5176
export JULIA_CUDA_MEMORY_POOL=none

srun -n $(Ngpu) --ntasks-per-node=4 -c 32 -G $(Ngpu) --cpu_bind=verbose,cores --gpu-bind=none \
    $(Base.julia_cmd()[1]) \
    --project=$(project_path) \
    $(run_file)
"""
end

generate_and_submit(perlmutter_submit_job_writer, perlmutter_config; caller_file=@__FILE__)
