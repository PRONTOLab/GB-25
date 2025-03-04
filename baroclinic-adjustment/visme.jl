using CairoMakie
using Oceananigans
using Printf

# Function to extract label from directory name
function extract_label(dir_name)
    m = match(r"GPU_(F\d+)_([\d_]+)", dir_name)
    @show m.captures[2]
    if m !== nothing
        fp_type = replace(m.captures[1], "F" => "FP") # Convert "F32" to "FP32"
        resolution = replace(m.captures[2], "__" => "/", "_" => "") # Convert "1__8" to "1/8"
        return @sprintf("GPU - %s - %s", fp_type, resolution)
    end
    return "Unknown"
end

# Get all directories in the current folder matching the pattern
all_dirs = filter(isdir, readdir())

# Keep only directories that match the naming pattern (adjust regex if needed)
matching_dirs = filter(d -> occursin(r"GPU_(F\d+)_\d+__\d+", d), all_dirs)

# Sort directories for consistent plotting order
sort!(matching_dirs)

# Initialize figure
pt = 4/3
fig = Figure(; size=(600, 500), fontsize=12pt)
ax = Axis(fig[1, 1], xlabel="Time", ylabel="Kinetic Energy", title="Kinetic Energy Evolution")

# Iterate over directories and plot data
for dir in matching_dirs
    m = match(r"GPU_(F\d+)", dir)
    prefix = "baroclinic_adjustment_" * replace(m.captures[1], "F" => "Float") # Extract FP type
    filename = joinpath(dir, prefix * "_kinetic_energy.jld2")

    if isfile(filename)
        Et = FieldTimeSeries(filename, "E")
        label = extract_label(dir)
        lines!(ax, Et.times, Et.data[1, 1, 1, :], linewidth=4, label=label)
    end
end

# Add legend
# axislegend(ax)
axislegend(ax, position = :lt)

# Display figure
display(fig)

save("numerics.png", fig)
