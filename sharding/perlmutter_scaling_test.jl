using Dates, Random

username = ENV["USER"]
account = "m4672"

exe_path = "sharded_baroclinic_instability.jl"
run_file = joinpath(@__DIR__, exe_path)
out_path = joinpath(ENV["SCRATCH"], "GB25", "$(string(now()))_$(randstring(4))")
project_path = dirname(@__DIR__)

mkpath(out_path)

@info "User: $(username); Project: $(account)"
@info "run_file=$(run_file)"
@info "Writing all output to: $(out_path)"

# run params
submit   = true
run_name = "r_react_"
time     = "00:40:00"
Ngpus    = [4, 8, 16, 32, 64]
type     = "weak"

gpus_per_node = 4

for Ngpu in Ngpus
    job_dir = joinpath(out_path, "ngpu=$(Ngpu)")
    mkpath(job_dir)

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

    sbatch_name = joinpath(job_dir, "submit.sh")
    cp(joinpath(@__DIR__, "launcher.sh"), joinpath(job_dir, "launcher.sh"))

    open(sbatch_name, "w") do io
        println(io,
                """
#!/bin/bash -l

#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpu-bind=none
#SBATCH --job-name="$job_name"
#SBATCH --time=$time
#SBATCH --nodes=$Nnodes
#SBATCH --account=$account
#SBATCH --output=$job_dir/%j.out
#SBATCH --error=$job_dir/%j.err

cd $job_dir

source /global/common/software/nersc9/julia/scripts/activate_beta.sh
ml load julia/1.10.8

export SBATCH_ACCOUNT=$account 
export SALLOC_ACCOUNT=$account
export JULIA_CUDA_MEMORY_POOL=none
export Ngpu=$Ngpu
export resolution_fraction=$resolution_fraction
export JULIA_DEBUG="Reactant,Reactant_jll"
export JULIA_DEPOT_PATH=\$SCRATCH/julia

srun -n $Nnodes -c 32 -G $Ngpu --cpu-bind=cores ./launcher.sh julia --project=$project_path -O0 $run_file 
                """)
    end

    if submit
        run(`sbatch $sbatch_name`)
        run(`squeue -u $username`)
    else
        @warn "job not submitted"
    end
end

