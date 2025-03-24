using Dates

username = "gwagner"

sbatch_params = Dict(
    "account"     => "g191",
    # "partition"   => "debug",
    # "uenv"        => "julia/24.9:v1",
    # "view"        => "julia",
)

exe_path = "sharded_baroclinic_instability.jl"

# run params
submit   = true
run_name = "r_react_"
time     = "00:40:00"
Ngpus    = [4, 8] #[4, 8, 12, 16]
type     = "weak"

gpus_per_node = 4

for Ngpu in Ngpus
    MPICH_GPU_SUPPORT_ENABLED = 1

    run_id   = string(run_name, "_",
                      Dates.format(now(), "ud"), "_",
                      details * "_ngpu", Ngpu)

    job_name = "scaling_test_$Ngpu"

    !isinteger(cbrt(Ngpu)) && (@warn "problem size is not cubic")
    Nnodes = ceil(Int, Ngpu / gpus_per_node)
    @assert (Ngpu % gpus_per_node == 0) || (Ngpu == 1)

    if type == "weak"
        resolution_fraction = 4Ngpu
    else
        resolution_fraction = 4 * Ngpus[1]
    end

    @info "executable name: $(exe_name)"
    @info "number of GPUs: $(Ngpu)"
    @info "number of nodes: $(Nnodes)"
    @info "number of GPUs per node: $(gpus_per_node)"

    sbatch_name = joinpath(run_dir, "submit.sh")
    proj_dir    = joinpath(run_dir, "../")

    open(sbatch_name, "w") do io
        println(io,
                """
                #!/bin/bash -l

                #SBATCH --job-name="$job_name"
                #SBATCH --output=$run_dir/slurm.%j.o
                #SBATCH --error=$run_dir/slurm.%j.e
                #SBATCH --time=$time
                #SBATCH --nodes=$Nnodes
                #SBATCH --ntasks=$Ngpu
                #SBATCH --gpus-per-node=$gpus_per_node
                #SBATCH --ntasks-per-node=$gpus_per_node
                #SBATCH --constraint=gpu
                #SBATCH --exclusive
                """)

        for (k, v) in sbatch_params
            println(io, "#SBATCH --$k=$v")
        end

        println(io,
                """
                export Ngpu=$Ngpu
                export resolution_fraction=$resolution_fraction
                export JULIA_DEBUG="Reactant,Reactant_jll"
                export MPICH_GPU_SUPPORT_ENABLED=$(MPICH_GPU_SUPPORT_ENABLED)

                ulimit -s unlimited

                alias julia='/capstor/scratch/cscs/gwagner/daint/juliaup/bin/julia'

                srun --preserve-env --gpu-bind=per_task:1 --cpu_bind=sockets bash unset_then_launch.sh
                """)
    end

    if submit
        run(`sbatch $sbatch_name`)
        run(`squeue -u $username`)
    else
        @warn "job not submitted"
    end
end

