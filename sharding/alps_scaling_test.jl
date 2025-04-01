include("common_submission_generator.jl")

function alps_submit_job_writer(cfg::JobConfig, job_name, Nnodes, job_dir, Ngpu,
                                resolution_fraction, project_path, run_file)

    """
#!/bin/bash -l

#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(cfg.time)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
#SBATCH --nodes=$(Nnodes)
#SBATCH --ntasks-per-node=$(cfg.tasks_per_node)
#SBATCH --gpus-per-node=$(cfg.gpus_per_node)
#SBATCH --constraint=gpu
#SBATCH --account=$(cfg.account)
##SBATCH --exclusive

export JULIA_CUDA_USE_COMPAT=false
export JULIA_CUDA_MEMORY_POOL=none
export MPICH_GPU_SUPPORT_ENABLED=1

export Ngpu=$(Ngpu)
export OCEANANIGANS_ARCHITECTURE=$(cfg.arch_kind)
export FLOAT_TYPE=$(cfg.float_type)
export resolution_fraction=$(resolution_fraction)
export JULIA_DEBUG="Reactant,Reactant_jll"
export JULIA_DEPOT_PATH=$(join(Base.DEPOT_PATH, ':'))
# export TF_CPP_MAX_VLOG_LEVEL=3
# export XLA_FLAGS="--xla_dump_to=$(job_dir)/xla_dump"

ulimit -s unlimited

# Setting `--cpu_bind` is explicitly discouraged:
# <https://eth-cscs.github.io/cscs-docs/guides/gb2025/#slurm>.
# We only set it to vrbose to record what's going on
srun --preserve-env --gpu-bind=per_task:1 --cpu_bind=verbose $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) --threads=$(cfg.cpus_per_task) -O0 $(run_file) 2>&1
"""
end


out_dir = @__DIR__

kw = (; 
    account       = "g191",
    username      = ENV["USER"],
    out_dir       = @__DIR__,
    submit        = true,
    run_name      = "",
    time          = "00:40:00",
    type          = "weak",
    gpus_per_node = 4,
    cpus_per_task = 16,
)

#config1 = JobConfig(arch_kind="ReactantState", tasks_per_node=1, Ngpus-[4, 8]; kw...)
configs = [
    JobConfig(arch_kind="ReactantState", tasks_per_node=1, Ngpus=[4, 8, 12]; kw...),
    JobConfig(arch_kind="GPU", tasks_per_node=4, Ngpus=[4, 8, 12]; kw...),
    # JobConfig(arch_kind="GPU", tasks_per_node=1, Ngpus=[1]; kw...),
    # JobConfig(arch_kind="ReactantState", tasks_per_node=1, Ngpus=[1]; kw...),
]

# run(`$(Base.julia_cmd()) --project -e 'using Pkg; Pkg.instantiate()'`)

for config in configs
    generate_and_submit(alps_submit_job_writer, config; caller_file=@__FILE__)
end

