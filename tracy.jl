using Tracy
using TracyProfiler_jll

port = 9000 + rand(1:1000)
p = Tracy.capture("abernathy.tracy"; port)
run(addenv(`$(Base.julia_cmd()) --project=. correctness/abernathey_channel.jl`,
            "TRACY_PORT" => string(port),
            "JULIA_WAIT_FOR_TRACY" => "1"))
wait(p)