using CairoMakie
using Oceananigans
using Printf

# User-defined prefix for filtering
prefix = "r_oa_Mar10"

# Function to extract label from directory name
function extract_label(dir_name)
    m = match(r"GPU_(F\d+)_([\d_]+)", dir_name)
    if m !== nothing
        fp_type = replace(m.captures[1], "F" => "FP") # Convert "F32" to "FP32"
        resolution = replace(m.captures[2], "__" => "/", "_" => "") # Convert "1__8" to "1/8"
        return @sprintf("GPU - %s - %s", fp_type, resolution)
    end
    return "Unknown"
end

function vis_evol(prefix; disp_fig=false, save_fig=false)
    # Get all directories in the current folder matching the pattern
    all_dirs = filter(isdir, readdir())

    # Keep only directories that match the naming pattern (adjust regex if needed)
    matching_dirs = filter(d -> startswith(d, prefix) && occursin(r"GPU_(F\d+)_\d+__\d+", d), all_dirs)

    # Sort directories for consistent plotting order
    sort!(matching_dirs)

    # Initialize figure
    pt = 4 / 3
    fig = Figure(; size=(600, 500), fontsize=12pt)
    ax = Axis(fig[1, 1], xlabel="Time", ylabel="Kinetic Energy", title="Kinetic Energy Evolution")

    # Iterate over directories and plot data
    for dir in matching_dirs
        m = match(r"GPU_(F\d+)", dir)
        pre = "baroclinic_adjustment_" * replace(m.captures[1], "F" => "Float") # Extract FP type
        filename = joinpath(dir, pre * "_kinetic_energy.jld2")

        if isfile(filename)
            Et = FieldTimeSeries(filename, "E")
            label = extract_label(dir)
            lines!(ax, Et.times, Et.data[1, 1, 1, :], linewidth=4, label=label)
        end
    end

    # Add legend
    axislegend(ax, position=:rb)

    # Display figure
    disp_fig && display(fig)

    # Save figure
    save_fig && save("numerics_" * prefix * ".png", fig)
    return
end

vis_evol(prefix; disp_fig=false, save_fig=true)

# Fields visu
dir = "baroclinic-adjustment"
rundir = prefix * "_GPU_F64_1__8_ngpu1_F7Eq_oa12_OK"
filename = joinpath(dir, rundir, "baroclinic_adjustment_Float64" * "_fields.jld2")

function vis_field(prefix, filename; obs=1, disp_fig=false, save_fig=false, save_anim=false)
    ζ_timeseries = FieldTimeSeries(filename, "ζ")
    n = Observable(obs)
    ζ_top = @lift interior(ζ_timeseries[$n], :, :, 1)

    xζ, yζ, zζ = nodes(ζ_timeseries)
    xζ = xζ ./ 1e3 # convert m -> km
    yζ = yζ ./ 1e3 # convert m -> km

    set_theme!(Theme(fontsize=24))
    fig = Figure(size=(1300, 1000))
    axζ = Axis(fig[1, 1], xlabel="x", ylabel="y", aspect=1)
    hm = heatmap!(axζ, xζ, yζ, ζ_top, colorrange=(-5e-5, 5e-5), colormap=:balance)
    Colorbar(fig[1, 2], hm, label="Surface ζ(x, y) (s⁻¹)")

    disp_fig && display(fig)
    save_fig && save("field_" * prefix * ".png", fig)

    if save_anim
        nframes = size(ζ_timeseries.data)[4]
        frames = 1:nframes

        record(fig, "movie_" * prefix * ".mp4", frames, framerate=8) do i
            n[] = i
        end
    end
    return
end

vis_field(prefix, filename; save_anim=true)
