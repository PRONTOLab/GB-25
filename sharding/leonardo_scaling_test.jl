using Dates, Random

username = ENV["USER"]
account = "try25_giordano"

exe_path = Base.ARGS[1]
run_file = joinpath(@__DIR__, exe_path)
if !isfile(run_file)
    error("File $(run_file) does not exist")
end
out_path = joinpath(@__DIR__, "runs", "$(string(now(UTC)))_$(randstring(4))")
project_path = dirname(@__DIR__)

mkpath(out_path)

@info "User: $(username); Project: $(account)"
@info "run_file=$(run_file)"
@info "Writing all output to: $(out_path)"

# run params
submit   = true
run_name = "r_react_"
time     = "00:40:00"
Ngpus    = (4, 8, 16, 32, 64)
type     = "weak"

gpus_per_node = 4

for Ngpu in Ngpus
    ngpu_string = lpad(Ngpu, 5, '0')
    job_dir = joinpath(out_path, "ngpu=$(ngpu_string)")
    mkpath(job_dir)

    run_id   = string(run_name, "_",
                      Dates.format(now(UTC), "ud"), "_ngpu",  ngpu_string)

    job_name = "scaling_test_$(ngpu_string)"

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

    sbatch_name = joinpath(job_dir, "submit.sh")
    cp(joinpath(@__DIR__, "launcher.sh"), joinpath(job_dir, "launcher.sh"))

    open(sbatch_name, "w") do io
        println(io,
                """
#!/bin/bash -l

#SBATCH --job-name="$(job_name)"
#SBATCH --time=$(time)
#SBATCH --nodes=$(Nnodes)
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:$(gpus_per_node)
#SBATCH --cpus-per-task=16
#SBATCH --partition boost_usr_prod
#SBATCH --account=$(account)
#SBATCH --output=$(job_dir)/%j.out
#SBATCH --error=$(job_dir)/%j.err
#SBATCH --qos=$(Nnodes <= 64 ? "normal" : "boost_qos_bprod")
#SBATCH --gpu-bind=none

# Make ultra sure this env var isn't set before loading CUDA module
export LD_LIBRARY_PATH=""
unset LD_LIBRARY_PATH
# export SBATCH_ACCOUNT=$(account)
# export SALLOC_ACCOUNT=$(account)
export JULIA_CUDA_MEMORY_POOL=none
export Ngpu=$(Ngpu)
export resolution_fraction=$(resolution_fraction)
export JULIA_DEBUG="Reactant,Reactant_jll"
export JULIA_DEPOT_PATH=$(join(Base.DEPOT_PATH, ':'))
# export TF_CPP_MAX_VLOG_LEVEL=3
# export XLA_FLAGS="--xla_dump_to=$(job_dir)/xla_dump"

module load cuda/12.3
srun --cpu-bind=verbose,cores $(job_dir)/launcher.sh $(Base.julia_cmd()[1]) --project=$(project_path) -O0 $(run_file)
                """)
    end

    if submit
        run(`sbatch $(sbatch_name)`)
        run(`squeue -u $(username)`)
    else
        @warn "job not submitted"
    end
end
