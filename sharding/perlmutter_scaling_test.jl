using Dates

username = "blaschke"
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
#SBATCH -q interactive
#SBATCH --gpu-bind=none
#SBATCH --job-name="$job_name"
#SBATCH --time=$time
#SBATCH --nodes=$Nnodes
#SBATCH --account=$account

source /global/common/software/nersc9/julia/scripts/activate_beta.sh
ml load julia/1.10.8

export SBATCH_ACCOUNT=$account 
export SALLOC_ACCOUNT=$account
export JULIA_CUDA_MEMORY_POOL=none

cat > launch.sh << EoF_s
#! /bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY
exec \$*
EoF_s
chmod +x launch.sh
""")

        runfile="~/NESAP/GB-25/sharding/sharded_baroclinic_instability.jl"

        println(io,
                """
export Ngpu=$Ngpu
export resolution_fraction=$resolution_fraction
export JULIA_DEBUG="Reactant,Reactant_jll"
export JULIA_DEPOT_PATH=\$SCRATCH/julia

srun -n $Nnodes -c 32 -G $Ngpu --cpu-bind=cores ./launch.sh julia --project -O0 $runfile 
                """)
    end

    if submit
        run(`sbatch $sbatch_name`)
        run(`squeue -u $username`)
    else
        @warn "job not submitted"
    end
end

