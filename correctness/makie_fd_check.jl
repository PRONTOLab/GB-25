using Oceananigans
ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using FileIO, JLD2
using Printf
using Statistics

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode

using GLMakie

graph_directory = "run_abernathy_model_ad_spinup4000000_8100steps_fdcheck/"

data = jldopen(graph_directory * "data_differences.jld2", "r")

Nx = data["Nx"]
Ny = data["Ny"]
Nz = data["Nz"]

wind_stress_ad_zonal = data["wind_stress_ad_zonal"]
wind_stress_ad_meridional = data["wind_stress_ad_meridional"]
wind_stress_differences_zonal = data["wind_stress_differences_zonal"]
wind_stress_differences_meridional = data["wind_stress_differences_meridional"]

close(data)

relative_error_zonal = abs.(wind_stress_ad_zonal .- wind_stress_differences_zonal) ./ abs.(wind_stress_ad_zonal)
relative_error_zonal = relative_error_zonal[1:2:Nx,:]
relative_error_zonal = minimum(relative_error_zonal; dims=2)

relative_error_meridional = abs.(wind_stress_ad_meridional .- wind_stress_differences_meridional) ./ abs.(wind_stress_ad_meridional)
relative_error_meridional = relative_error_meridional[1:2:Ny,:]
relative_error_meridional = minimum(relative_error_meridional; dims=2)

relative_error_zonal = vec(relative_error_zonal')
relative_error_meridional = vec(relative_error_meridional')

@show relative_error_zonal
@show relative_error_meridional

scale = 1000 / Nx

fig1 = Figure(size = (2000, 1000))
ax1 = Axis(fig1[1, 1],
          yscale = log10,
          #xlabel = "Zonal Position (kilometers)",
          ylabel = "Relative Error",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

# Here we'll plot magnitudes of gradients for reference:
wind_stress_ad_zonal = vec(wind_stress_ad_zonal[1:2:Nx,:]')
wind_stress_ad_meridional = vec(wind_stress_ad_meridional[1:2:Ny,:]')

ax2 = Axis(fig1[2, 1],
          yscale = log10,
          xlabel = "Zonal Position (kilometers)",
          ylabel = "Magnitude of Derivative",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

scatter!(ax1, collect(1:2:Nx), relative_error_zonal; markersize = 20)
scatter!(ax2, collect(1:2:Nx), abs.(wind_stress_ad_zonal); markersize = 20)
save(graph_directory * "wind_stress_difference_zonal.png", fig1)

fig2 = Figure(size = (2000, 1000))
ax1 = Axis(fig2[1, 1],
          yscale = log10,
          #xlabel = "Meridional Position (kilometers)",
          ylabel = "Relative Error",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

ax2 = Axis(fig2[2, 1],
          yscale = log10,
          xlabel = "Meridional Position (kilometers)",
          ylabel = "Magnitude of Derivative",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

scatter!(ax1, collect(1:2:Ny), relative_error_meridional; markersize = 20)
scatter!(ax2, collect(1:2:Ny), abs.(wind_stress_ad_meridional); markersize = 20)
save(graph_directory * "wind_stress_difference_meridional.png", fig2)




graph_directory = "run_abernathy_model_ad_spinup4000000_8100steps_fdcheck_temp/"

data = jldopen(graph_directory * "data_differences.jld2", "r")

Nx = data["Nx"]
Ny = data["Ny"]
Nz = data["Nz"]

temp_ad_zonal = data["temp_ad_zonal"]
temp_ad_meridional = data["temp_ad_meridional"]
temp_differences_zonal = data["temp_differences_zonal"]
temp_differences_meridional = data["temp_differences_meridional"]

close(data)

relative_error_zonal = abs.(temp_ad_zonal .- temp_differences_zonal) ./ abs.(temp_ad_zonal)
relative_error_zonal = relative_error_zonal[1:2:Nx,:]
relative_error_zonal = minimum(relative_error_zonal; dims=2)

relative_error_meridional = abs.(temp_ad_meridional .- temp_differences_meridional) ./ abs.(temp_ad_meridional)
relative_error_meridional = relative_error_meridional[1:2:Ny,:]
relative_error_meridional = minimum(relative_error_meridional; dims=2)

relative_error_zonal = vec(relative_error_zonal')
relative_error_meridional = vec(relative_error_meridional')

@show relative_error_zonal
@show relative_error_meridional

scale = 1000 / Nx

fig1 = Figure(size = (2000, 1000))
ax1 = Axis(fig1[1, 1],
          yscale = log10,
          #xlabel = "Zonal Position (kilometers)",
          ylabel = "Relative Error",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

# Here we'll plot magnitudes of gradients for reference:
temp_ad_zonal = vec(temp_ad_zonal[1:2:Nx,:]')
temp_ad_meridional = vec(temp_ad_meridional[1:2:Ny,:]')

ax2 = Axis(fig1[2, 1],
          yscale = log10,
          xlabel = "Zonal Position (kilometers)",
          ylabel = "Magnitude of Derivative",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

scatter!(ax1, collect(1:2:Nx), relative_error_zonal; markersize = 20)
scatter!(ax2, collect(1:2:Nx), abs.(temp_ad_zonal); markersize = 20)
save(graph_directory * "temp_difference_zonal.png", fig1)

fig2 = Figure(size = (2000, 1000))
ax1 = Axis(fig2[1, 1],
          yscale = log10,
          #xlabel = "Meridional Position (kilometers)",
          ylabel = "Relative Error",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

ax2 = Axis(fig2[2, 1],
          yscale = log10,
          xlabel = "Meridional Position (kilometers)",
          ylabel = "Magnitude of Derivative",
          titlesize = 24,
          xlabelsize = 24,
          xticklabelsize = 20,
          xtickformat = values -> ["$(value*scale)" for value in values],
          ylabelsize = 24,
          yticklabelsize = 20)

scatter!(ax1, collect(1:2:Ny), relative_error_meridional; markersize = 20)
scatter!(ax2, collect(1:2:Ny), abs.(temp_ad_meridional); markersize = 20)
save(graph_directory * "temp_difference_meridional.png", fig2)