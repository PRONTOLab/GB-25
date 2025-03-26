using Dates, Random

username = ENV["USER"]

@kwdef struct JobConfig
    username::String
    account::String
    out_dir::String
    time::String
    cpus_per_task::Int
    Ngpus::Vector{Int}
    run_name::String
    gpus_per_node::Int
    type::String
    submit::Bool
end

function generate_and_submit(submit_job_writer, cfg::JobConfig; input_file_name::String)

    if !isone(length(Base.ARGS))
        error("""
              Usage:

                  julia $(basename(input_file_name)) <SCRIPT_PATH>

              E.g.:

                  julia $(basename(input_file_name)) sharded_baroclinic_instability.jl
              """)
    end

    exe_path = Base.ARGS[1]
    run_file = joinpath(@__DIR__, exe_path)
    if !isfile(run_file)
        error("File $(run_file) does not exist")
    end
    # Some filesystems don't like colons in directory names
    timestamp = replace(string(now(UTC)), ':' => '-')
    out_path = joinpath(cfg.out_dir, "runs", "$(timestamp)_$(randstring(4))")
    project_path = dirname(@__DIR__)

    mkpath(out_path)

    @info "User: $(cfg.username); Project: $(cfg.account)"
    @info "run_file=$(run_file)"
    @info "Writing all output to: $(out_path)"

    for Ngpu in cfg.Ngpus
        ngpu_string = lpad(Ngpu, 5, '0')
        job_dir = joinpath(out_path, "ngpu=$(ngpu_string)")
        mkpath(job_dir)

        run_id   = string(run_name, "_",
                          Dates.format(now(UTC), "ud"), "_ngpu",  ngpu_string)

        job_name = "scaling_test_$(ngpu_string)"

        # !isinteger(cbrt(Ngpu)) && (@warn "problem size is not cubic")
        Nnodes = ceil(Int, Ngpu / cfg.gpus_per_node)
        @assert (Ngpu % cfg.gpus_per_node == 0) || (Ngpu == 1)

        if cfg.type == "weak"
            resolution_fraction = 4Ngpu
        else
            resolution_fraction = 4 * cfg.Ngpus[1]
        end

        @info "number of GPUs: $(Ngpu)"
        @info "number of nodes: $(Nnodes)"
        @info "number of GPUs per node: $(cfg.gpus_per_node)"

        sbatch_name = joinpath(job_dir, "submit.sh")

        launcher = joinpath(job_dir, "launcher.sh")
        open(launcher, "w") do io
            print(io, """
#!/usr/bin/env sh

export CUDA_VISIBLE_DEVICES=$(join(0:(min(Ngpu, gpus_per_node) - 1), ','))

# Important else XLA might hang indefinitely
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

exec "\${@}"
echo "[\${SLURM_JOB_ID}.\${SLURM_PROCID}] Process exited with code \${?}"
""")
        end
        chmod(launcher, 0o755)

        open(sbatch_name, "w") do io
            print(io, submit_job_writer(cfg::JobConfig, job_name::String,
                                        Nnodes::Int, job_dir::String, Ngpu::Int,
                                        resolution_fraction::Int,
                                        project_path::String, run_file::String))
        end
        if cfg.submit
            run(`sbatch $(sbatch_name)`)
            run(`squeue -u $(cfg.username)`)
        else
            @warn "job not submitted"
        end
    end
end
