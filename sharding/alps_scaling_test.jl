using Dates

submit        = false
username      = "gwagner"
run_name      = "reactant_"
time          = "00:40:00"
Ngpus         = [4, 8] #[4, 8, 12, 16]
type          = "weak"
sbatch_prefix = "submit"
gpus_per_node = 4

sbatch_params = Dict(
    "account"     => "g191",
    # "partition"   => "debug",
    # "uenv"        => "julia/24.9:v1",
    # "view"        => "julia",
)

##SBATCH --ntasks=$Nnodes

for Ngpu in Ngpus
    MPICH_GPU_SUPPORT_ENABLED = 1
    run_id   = string(run_name, "_", Dates.format(now(), "ud"), "_ngpu", Ngpu)
    job_name = "scaling_test_$Ngpu"

    !isinteger(cbrt(Ngpu)) && (@warn "problem size is not cubic")
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

    sbatch_name = string(sbatch_prefix, "_", Ngpu, ".sh")

    open(sbatch_name, "w") do io
        println(io,
                """
#!/bin/bash -l

#SBATCH --job-name="$job_name"
#SBATCH --output=slurm.%j.o
#SBATCH --error=slurm.%j.e
#SBATCH --time=$time
#SBATCH --gpus-per-node=$gpus_per_node
#SBATCH --nodes=$Nnodes
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --exclusive""")

        for (k, v) in sbatch_params
            println(io, "#SBATCH --$k=$v")
        end

        println(io,
                """

export Ngpu=$Ngpu
export resolution_fraction=$resolution_fraction
export JULIA_DEBUG="Reactant,Reactant_jll"
export MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED

cat > launch.sh << EoF_s
#! /bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Important else XLA might hang indefinitely
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY
exec \$*
EoF_s
chmod +x launch.sh

ulimit -s unlimited
alias julia='/capstor/scratch/cscs/gwagner/daint/juliaup/bin/julia'
srun --preserve-env --gpu-bind=per_task:1 --cpu_bind=sockets ./launch.sh \\
    julia --project --threads=auto -O0 \${HOME}/GB-25/sharding/sharded_baroclinic_instability.jl
                """)
    end

    if submit
        run(`sbatch $sbatch_name`)
        run(`squeue -u $username`)
    else
        @warn "job not submitted"
    end
end

