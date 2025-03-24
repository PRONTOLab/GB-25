using Dates

username = "ssilvest"
account = "m4672"

exe_path = "sharded_baroclinic_instability.jl"

# run params
submit   = true
run_name = "r_react_"
time     = "00:40:00"
Ngpus    = [4, 8, 16, 32, 64]
type     = "weak"

gpus_per_node = 4

for Ngpu in Ngpus
    MPICH_GPU_SUPPORT_ENABLED = 1

    run_id   = string(run_name, "_",
                      Dates.format(now(), "ud"), "_ngu",  Ngpu)

    job_name = "scaling_test_$Ngpu"

    Nnodes = ceil(Int, Ngpu / gpus_per_node)
    @assert (Ngpu % gpus_per_node == 0) || (Ngpu == 1)

    if type == "weak"
        resolution_fraction = 4Ngpu
    else
        resolution_fraction = 4 * Ngpus[1]
    end

    @info "number of GPUs: $(Ngpu)"
    @info "number of nodes: $(Nnodes)"
    @info "number of GPUs per node: $(gpus_per_node)"

    sbatch_name = "./submit.sh"

    open(sbatch_name, "w") do io
        println(io,
                """
#!/bin/bash -l

#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -c 32
#SBATCH --gpu-bind=none
#SBATCH --job-name="$job_name"
#SBATCH --output=slurm.%j.o
#SBATCH --error=slurm.%j.e
#SBATCH --time=$time
#SBATCH --nodes=$Nnodes
#SBATCH --ntasks=$Ngpu
#SBATCH --gpus-per-node=$gpus_per_node
#SBATCH --ntasks-per-node=$gpus_per_node
#SBATCH --account=$account
module load cray-mpich

export SBATCH_ACCOUNT=$account 
export SALLOC_ACCOUNT=$account
export JULIA_CUDA_MEMORY_POOL=none

export SLURM_CPU_BIND="cores"
export CRAY_ACCEL_TARGET="nvidia80"

cat > launch.sh << EoF_s
#! /bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec \$*
EoF_s
chmod +x launch.sh

""")

        runfile="/pscratch/sd/s/ssilvest/GB-25/sharding/sharded_baroclinic_instability.jl"

        println(io,
                """
export Ngpu=$Ngpu
export resolution_fraction=$resolution_fraction
export JULIA_DEBUG="Reactant,Reactant_jll"
export MPICH_GPU_SUPPORT_ENABLED=$(MPICH_GPU_SUPPORT_ENABLED)

alias julia='/global/homes/s/ssilvest/julia-1.10.9/bin/julia'

srun --preserve-env --gpu-bind=per_task:1 --cpu_bind=sockets ./launch.sh julia --project -O0 $runfile 
                """)
    end

    if submit
        run(`sbatch $sbatch_name`)
        run(`squeue -u $username`)
    else
        @warn "job not submitted"
    end
end

