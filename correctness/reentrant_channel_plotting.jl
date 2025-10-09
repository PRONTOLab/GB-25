using FileIO, JLD2
using GLMakie

#
# For plotting and postprocessing.
#


graph_directory = "run_abernathy_model_ad_900steps_noCATKE_mildVisc_CenteredOrder4_partialCell/"

data1 = jldopen(graph_directory * "data_init.jld2", "r")

Nx = data1["Nx"]
Ny = data1["Ny"]
Nz = data1["Nz"]

bottom_height = data1["bottom_height"]
T_init        = data1["T_init"]
e_init        = data1["e_init"]
wind_stress   = data1["u_wind_stress"]
#=
# Build init temperature fields:
max_T_surface = maximum(abs.(T_init[1:Nx,1:Ny,Nz]))

fig, ax, hm = heatmap(view(T_init, 1:Nx, 1:Ny, Nz),
                                colormap = :thermal,
                                #colorrange = (-max_T_surface, max_T_surface),
                                axis = (xlabel = "x [indices]",
                                        ylabel = "y [indices]",
                                        title = "T(x, y, z=0, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_surface.png", fig)


max_T_deep = maximum(abs.(T_init[1:Nx,1:Ny,1]))

fig, ax, hm = heatmap(view(T_init, 1:Nx, 1:Ny, 1),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "x [indices]",
                                        ylabel = "y [indices]",
                                        title = "T(x, y, z=-4000, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_bottom.png", fig)

max_T_cross = maximum(abs.(T_init[1:Nx,20,1:Nz]))

fig, ax, hm = heatmap(view(T_init, 1:Nx, 20, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "x [indices]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x, y=-50, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_zonal.png", fig)

max_T_cross = maximum(abs.(T_init[140,1:Ny,1:Nz]))

fig, ax, hm = heatmap(view(T_init, 140, 1:Ny, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "y [indices]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x=140, y, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

resize_to_layout!(fig)
save(graph_directory * "init_T_meridional.png", fig)
=#
@info "Plotted initial T"


# Ridge:
fig, ax, hm = heatmap(view(bottom_height, 1:Nx, 1:Ny),
                      colormap = :deep,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "bottom height(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m")

save(graph_directory * "bottom_height.png", fig)
#=
# Energy:
fig, ax, hm = heatmap(view(e_init, 1:Nx, 1:Ny, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "e(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

resize_to_layout!(fig)
save(graph_directory * "init_e_surface.png", fig)

fig, ax, hm = heatmap(view(e_init, 1:Nx, 1:Ny, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "e(x, y, z=-4000, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

resize_to_layout!(fig)
save(graph_directory * "init_e_bottom.png", fig)


# Wind stress:
fig, ax, hm = heatmap(view(wind_stress, 1:Nx, 1:Ny),
                      colormap = :deep,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "zonal wind_stress(x, y, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

resize_to_layout!(fig)
save(graph_directory * "init_wind_stress.png", fig)
=#
close(data1)

#
# Data from end of run:
#
#=
data = jldopen(graph_directory * "data_final.jld2", "r")

Nx = data["Nx"]
Ny = data["Ny"]
Nz = data["Nz"]

T_final = data["T_final"]
e_final = data["e_final"]
ssh     = data["ssh"]
mld     = data["mld"]

u = data["u"]
v = data["v"]
w = data["w"]

dwind_stress = data["dwind_stress"]
dT           = data["dT"]

max_surface_T = maximum(abs.(T_final[:,:,Nz]))

# Build final temperature fields:
fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, Nz),
                      colormap = :thermal,
                      colorrange = (0, max_surface_T),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "T(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

save(graph_directory * "final_T_surface.png", fig)


fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, 32),
                      colormap = :thermal,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "T(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

save(graph_directory * "final_T_200.png", fig)

fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, 16),
                      colormap = :thermal,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "T(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

save(graph_directory * "final_T_1000.png", fig)


max_bottom_T = maximum(abs.(T_final[:,:,1]))

fig, ax, hm = heatmap(view(T_final, 1:Nx, 1:Ny, 1),
                      colormap = :thermal,
                      colorrange = (0, max_bottom_T),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "T(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

save(graph_directory * "final_T_bottom.png", fig)
#=
max_T_cross = maximum(abs.(T_final[1:Nx,20,1:Nz]))

fig, ax, hm = heatmap(view(T_final, 1:Nx, 20, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "x [indices]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x, y=-50, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

resize_to_layout!(fig)
save(graph_directory * "final_T_zonal.png", fig)

max_T_cross = maximum(abs.(T_final[140,1:Ny,1:Nz]))

fig, ax, hm = heatmap(view(T_final, 140, 1:Ny, 1:Nz),
                                        colormap = :thermal,
                                        #colorrange = (-max_T_deep, max_T_deep),
                                        axis = (xlabel = "y [indices]",
                                        ylabel = "z [indices - not evenly spaced]",
                                        title = "T(x=140, y, z, t=0)",
                                        titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[indices C]")

resize_to_layout!(fig)
save(graph_directory * "final_T_meridional.png", fig)
=#
#
# Energy
#
fig, ax, hm = heatmap(view(e_final, 1:Nx, 1:Ny, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "e(x, y, z=0, end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save(graph_directory * "final_e_surface.png", fig)

fig, ax, hm = heatmap(view(e_final, 1:Nx, 1:Ny, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "e(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save(graph_directory * "final_e_bottom.png", fig)


# Vertical velocity:
max_surface_w = maximum(abs.(w[1:Nx, 1:Ny, Nz]))

fig, ax, hm = heatmap(view(w, 1:Nx, 1:Ny, Nz),
                      colormap = :seismic,
                      colorrange = (-max_surface_w, max_surface_w),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "w(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_surface_w.png", fig)

# Zonal velocity:
max_surface_u = maximum(abs.(u[1:Nx, 1:Ny, Nz]))

fig, ax, hm = heatmap(view(u, 1:Nx, 1:Ny, Nz),
                      colormap = :seismic,
                      colorrange = (-max_surface_u, max_surface_u),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "u(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_surface_u.png", fig)

max_200_u = maximum(abs.(u[1:Nx, 1:Ny, 16]))

fig, ax, hm = heatmap(view(u, 1:Nx, 1:Ny, 16),
                      colormap = :seismic,
                      colorrange = (-max_surface_u, max_surface_u),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "u(x, y, z=200m, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_200_u.png", fig)

max_deep_u = maximum(abs.(u[1:Nx, 1:Ny, 1]))

fig, ax, hm = heatmap(view(u, 1:Nx, 1:Ny, 1),
                      colormap = :seismic,
                      colorrange = (-max_deep_u, max_deep_u),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "u(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_deep_u.png", fig)

# Meridional velocity:
max_surface_v = maximum(abs.(v[1:Nx, 1:Ny, Nz]))

fig, ax, hm = heatmap(view(v, 1:Nx, 1:Ny, Nz),
                      colormap = :seismic,
                      colorrange = (-max_surface_v, max_surface_v),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "v(x, y, z=0, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_surface_v.png", fig)

fig, ax, hm = heatmap(view(v, 1:Nx, 1:Ny, 16),
                      colormap = :seismic,
                      colorrange = (-max_surface_u, max_surface_u),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "v(x, y, z=200m, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_200_v.png", fig)

max_deep_v = maximum(abs.(v[1:Nx, 1:Ny, 1]))

fig, ax, hm = heatmap(view(v, 1:Nx, 1:Ny, 1),
                      colormap = :seismic,
                      colorrange = (-max_deep_v, max_deep_v),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "v(x, y, z=-4000, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save(graph_directory * "final_deep_v.png", fig)

max_η = maximum(abs.(ssh))

fig, ax, hm = heatmap(view(ssh, 1:Nx, 1:Ny, 1),
                      colormap = :seismic,
                      colorrange = (-max_η, max_η),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "SSH(x, y, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m")

save(graph_directory * "final_SSH.png", fig)
#=
max_mld = maximum(mld)
min_mld = minimum(mld)

fig, ax, hm = heatmap(view(mld, 1:Nx, 1:Ny, 1),
                      colormap = :deep,
                      colorrange = (min_mld, max_mld),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "mld(x, y, t=end)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m")

save(graph_directory * "final_mld.png", fig)
=#


max_dwind_stress = maximum(dwind_stress)
min_dwind_stress = minimum(dwind_stress)

fig, ax, hm = heatmap(view(dwind_stress, 1:Nx, 1:Ny, 1),
                      colormap = :deep,
                      colorrange = (min_dwind_stress, max_dwind_stress),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "dwind_stress(x, y)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m")

save(graph_directory * "dwind_stress.png", fig)

max_dT = maximum(dT)
min_dT = minimum(dT)

fig, ax, hm = heatmap(view(dT, 1:Nx, 1:Ny, 1),
                      colormap = :deep,
                      colorrange = (min_dT, max_dT),
                      axis = (xlabel = "x [indices]",
                              ylabel = "y [indices]",
                              title = "dT(x, y)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "m")

save(graph_directory * "dT.png", fig)

close(data)
=#