using FileIO, JLD2
using GLMakie

#
# For plotting and postprocessing.
#


graph_directory = "gpu_abernathy_runs/run_abernathy_model_1000days_notReactant/"

data = jldopen(graph_directory * "data_init.jld2", "r")

Nx = data["Nx"]
Ny = data["Ny"]
Nz = data["Nz"]

bottom_height = data["bottom_height"]
T_init        = data["b_init"]
e_init        = data["e_init"]
wind_stress   = data["wind_stress"]

# Build init temperature fields:
max_T_surface = maximum(abs.(T_init[1:Nx,1:Ny,Nz]))

fig, ax, hm = heatmap(view(T_init, 1:Nx, 1:Ny, Nz),
                                colormap = :thermal,
                                #colorrange = (-max_T_surface, max_T_surface),
                                axis = (xlabel = "x [degrees]",
                                        ylabel = "y [degrees]",
                                        title = "T(x, y, z=0, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_surface.png", fig)


max_T_deep = maximum(abs.(T_init[1:Nx,1:Ny,1]))

fig, ax, hm = heatmap(view(T_init, 1:Nx, 1:Ny, 1),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "x [degrees]",
                                        ylabel = "y [degrees]",
                                        title = "T(x, y, z=-4000, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_bottom.png", fig)
#=
max_T_cross = maximum(abs.(T_init[1:Nx,20,1:Nz]))

fig, ax, hm = heatmap(view(T_init, 1:Nx, 20, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "x [degrees]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x, y=-50, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_zonal.png", fig)

max_T_cross = maximum(abs.(T_init[140,1:Ny,1:Nz]))

fig, ax, hm = heatmap(view(T_init, 140, 1:Ny, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "y [degrees]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x=140, y, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_meridional.png", fig)
=#
@info "Plotted initial T"


# Ridge:
fig, ax, hm = heatmap(view(bottom_height, 1:Nx, 1:Ny),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "bottom height(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m")

save(graph_directory * "bottom_height.png", fig)

# Energy:
fig, ax, hm = heatmap(view(e_init, 1:Nx, 1:Ny, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

resize_to_layout!(fig)
save(graph_directory * "init_e_surface.png", fig)

fig, ax, hm = heatmap(view(e_init, 1:Nx, 1:Ny, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=-4000, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

resize_to_layout!(fig)
save(graph_directory * "init_e_bottom.png", fig)


# Wind stress:
fig, ax, hm = heatmap(view(wind_stress, 1:Nx, 1:Ny),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "zonal wind_stress(x, y, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

resize_to_layout!(fig)
save(graph_directory * "init_wind_stress.png", fig)

close(data)

#
# Data from end of run:
#

data = jldopen(graph_directory * "data_final.jld2", "r")

Nx = data["Nx"]
Ny = data["Ny"]
Nz = data["Nz"]

T_final = data["b_final"]
e_final = data["e_final"]
ssh     = data["ssh"]

u = data["u"]
v = data["v"]
w = data["w"]

# Build final temperature fields:
fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, Nz),
                      colormap = :thermal,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save(graph_directory * "final_T_surface.png", fig)

fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, 32),
                      colormap = :thermal,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save(graph_directory * "final_T_200.png", fig)

fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, 16),
                      colormap = :thermal,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save(graph_directory * "final_T_1000.png", fig)

fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, 1),
                      colormap = :thermal,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save(graph_directory * "final_T_bottom.png", fig)
#=
max_T_cross = maximum(abs.(T_final[1:Nx,20,1:Nz]))

fig, ax, hm = heatmap(view(T_final, 1:Nx, 20, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "x [degrees]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x, y=-50, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

resize_to_layout!(fig)
save(graph_directory * "final_T_zonal.png", fig)

max_T_cross = maximum(abs.(T_final[140,1:Ny,1:Nz]))

fig, ax, hm = heatmap(view(T_final, 140, 1:Ny, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "y [degrees]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x=140, y, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

resize_to_layout!(fig)
save(graph_directory * "final_T_meridional.png", fig)
=#
#
# Energy
#
fig, ax, hm = heatmap(view(e_final, 1:Nx, 1:Ny, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=0, end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save(graph_directory * "final_e_surface.png", fig)

fig, ax, hm = heatmap(view(e_final, 1:Nx, 1:Ny, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save(graph_directory * "final_e_bottom.png", fig)


# Vertical velocity:
max_surface_w = maximum(abs.(w[1:Nx, 1:Ny, Nz]))

fig, ax, hm = heatmap(view(w, 1:Nx, 1:Ny, Nz),
                      colormap = :seismic,
                      colorrange = (-max_surface_w, max_surface_w),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "w(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_surface_w.png", fig)

# Zonal velocity:
max_surface_u = maximum(abs.(u[1:Nx, 1:Ny, Nz]))

fig, ax, hm = heatmap(view(u, 1:Nx, 1:Ny, Nz),
                      colormap = :seismic,
                      colorrange = (-max_surface_u, max_surface_u),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "u(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_surface_u.png", fig)

max_200_u = maximum(abs.(u[1:Nx, 1:Ny, 16]))

fig, ax, hm = heatmap(view(u, 1:Nx, 1:Ny, 16),
                      colormap = :seismic,
                      colorrange = (-max_surface_u, max_surface_u),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "u(x, y, z=200m, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_200_u.png", fig)

max_deep_u = maximum(abs.(u[1:Nx, 1:Ny, 1]))

fig, ax, hm = heatmap(view(u, 1:Nx, 1:Ny, 1),
                      colormap = :seismic,
                      colorrange = (-max_deep_u, max_deep_u),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "u(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_deep_u.png", fig)

# Meridional velocity:
max_surface_v = maximum(abs.(v[1:Nx, 1:Ny, Nz]))

fig, ax, hm = heatmap(view(v, 1:Nx, 1:Ny, Nz),
                      colormap = :seismic,
                      colorrange = (-max_surface_v, max_surface_v),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "v(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_surface_v.png", fig)

fig, ax, hm = heatmap(view(v, 1:Nx, 1:Ny, 16),
                      colormap = :seismic,
                      colorrange = (-max_surface_u, max_surface_u),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "v(x, y, z=200m, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_200_v.png", fig)

max_deep_v = maximum(abs.(v[1:Nx, 1:Ny, 1]))

fig, ax, hm = heatmap(view(v, 1:Nx, 1:Ny, 1),
                      colormap = :seismic,
                      colorrange = (-max_deep_v, max_deep_v),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "v(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_deep_v.png", fig)

max_η = maximum(abs.(ssh))

fig, ax, hm = heatmap(view(ssh, 1:Nx, 1:Ny, 1),
                      colormap = :seismic,
                      colorrange = (-max_η, max_η),
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "SSH(x, y, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m")

save(graph_directory * "final_SSH.png", fig)

close(data)