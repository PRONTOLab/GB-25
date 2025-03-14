using Dates, Random, JSON

my_uuid  = randstring(4)
username = "lraess"

sbatch_params = Dict(
    "account"     => "c44",
    # "partition"   => "debug",
    # "uenv"        => "julia/24.9:v1",
    # "view"        => "julia",
)

exe_name = "baroclinic_adjustment.jl"
exe_subdir = "baroclinic-adjustment"

# input params
params = Dict(
    "precis" => "Float64", # Float64, Float32
    "arch"   => "GPU()",   # CPU(), GPU(), ReactantState()
    "resol"  => "1//16"     # 1//8, 1//16
)

details = replace(params["arch"] * "_" * params["precis"] * "_" * params["resol"], "Float" => "F", "//" => "__", "()" => "")

# run params
submit   = true
run_name = "r_oa"
time     = "00:20:00"
# num_gpus = 4 * 2 # 1 | 2, 16, 128, 480, 600, 1024, 2000, 2662, 3456, 4394, 5488
num_gpus_vec = 4 .* (2, 16, 128)

for num_gpus in (1, #= num_gpus_vec... =#)
    gpus_per_node = 4
    MPICH_GPU_SUPPORT_ENABLED = 1

    # gen run ID and create run folder
    # run_id   = run_name * "_" * Dates.format(now(),"ud") * "_ngpu" * string(num_gpus) * "_" * my_uuid
    run_id   = run_name * "_" * Dates.format(now(),"ud") * "_" * details * "_ngpu" * string(num_gpus) * "_" * my_uuid
    job_name = "OA_" * my_uuid

    !isinteger(cbrt(num_gpus)) && (@warn "problem size is not cubic")
    num_nodes = ceil(Int, num_gpus / gpus_per_node)
    @assert (num_gpus % gpus_per_node == 0) || (num_gpus == 1)

    @info "executable name: $(exe_name)"
    @info "number of GPUs: $(num_gpus)"
    @info "number of nodes: $(num_nodes)"
    @info "number of GPUs per node: $(gpus_per_node)"

    # Job dir and job file creation
    run_dir = joinpath(@__DIR__, run_id)
    @info "run dir: $(run_dir)"

    mkdir(run_dir)
    exe_path = joinpath(exe_subdir, exe_name)
    run(`cp $exe_path $run_dir`)

    params_name = joinpath(run_dir, "params.json")
    sbatch_name = joinpath(run_dir, "submit.sh")
    proj_dir    = joinpath(run_dir, "../")

    open(io -> JSON.print(io, params), params_name, "w")

    open(sbatch_name, "w") do io
        println(io,
                """
                #!/bin/bash -l

                #SBATCH --job-name="$job_name"
                #SBATCH --output=$run_dir/slurm.%j.o
                #SBATCH --error=$run_dir/slurm.%j.e
                #SBATCH --time=$time
                #SBATCH --nodes=$num_nodes
                #SBATCH --ntasks=$num_gpus
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

                export MPICH_GPU_SUPPORT_ENABLED=$(MPICH_GPU_SUPPORT_ENABLED)

                srun --gpu-bind=per_task:1 --cpu_bind=sockets julia --project=$proj_dir --color=yes $(joinpath(run_dir, exe_name))
                """)
    end

    if submit
        run(`sbatch $sbatch_name`)
        run(`squeue -u $username`)
    else
        @warn "job not submitted"
    end
end
