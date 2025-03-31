using GLMakie
using Statistics
using JLD2

function benchseries(filename, FT)
    file = jldopen(filename, "r")
    steps = Float64[]
    αs = keys(file[FT])
    for α in αs
        mean_step = Statistics.mean(file["$FT/$α/step"])
        push!(steps, mean_step)
    end
    return steps
end

filename = "timings_CPU.jld2"
f64times = benchseries(filename, "Float64")
f32times = benchseries(filename, "Float32")

α = 1:22
N² = 16 * 8
Nh = sqrt.(α.^2 * N²)

fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[2, 1])
scatterlines!(ax1, Nh .^2, f64times ./ 10) 
scatterlines!(ax2, Nh .^2, f64times ./ (10 * 128 * Nh.^2))

scatterlines!(ax1, Nh .^2, f32times[1:22] ./ 10) 
scatterlines!(ax2, Nh .^2, f32times[1:22] ./ (10 * 128 * Nh.^2))

