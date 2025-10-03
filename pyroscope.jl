p = run(pipeline(addenv(`$(Base.julia_cmd()) --threads=4,4 --project=. correctness/abernathey_channel.jl`,
            "JULIA_PPROF" => "1"), stdout=stdout, stderr=stderr), wait=false)


sleep(10)

while Base.process_running(p)
    run(`curl '127.0.0.1:16825/profile?seconds=10' --output prof_10s.pb.gz`)
    run(`./profilecli upload --override-timestamp prof_10s.pb.gz`) 
end

# pyroscope  -validation.max-profile-size-bytes 0 -validation.max-profile-stacktrace-depth 0 -validation.max-profile-stacktrace-samples 0 -distributor.ingestion-rate-limit-mb 1024 -distributor.ingestion-burst-size-mb 32