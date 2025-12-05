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

graph_directory1 = "run_abernathy_model_ad_spinup10000_100steps_lowRes_noImmersedGrid_noRandomT_lowerVisc_v101p3/"
graph_directory2 = "run_abernathy_model_ad_spinup10000_100steps_lowRes_zeroImmersedGrid_noRandomT_lowerVisc_v101p3/"

#graph_directory1 = "run_abernathy_model_ad_spinup4000000_8100steps_noImmersedBC/"
#graph_directory2 = "run_abernathy_model_ad_spinup4000000_8100steps_zeroImmersedBC/"

#
# First we gather the data and create a grid for plotting purposes:
#
data1init = jldopen(graph_directory1 * "data_init.jld2", "r")

Nx = data1init["Nx"]
Ny = data1init["Ny"]
Nz = data1init["Nz"]

# stretched grid
k_center = collect(1:Nz)
Δz_center = @. 10 * 1.104^(Nz - k_center)

halo_size = 4 #3 for non-immersed grid

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = sum(Δz_center)

z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
z_faces[Nz+1] = 0


function wall_function(x, y)
    zonal = (x > 470kilometers) && (x < 530kilometers)
    gap   = (y < 400kilometers) || (y > 1000kilometers)
    return (Lz+1) * zonal * gap - Lz
end


function make_grid(architecture, Nx, Ny, Nz, z_faces)

    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    # Make into a ridge array:
    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, wall_function)

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))
    return grid
end

grid = make_grid(CPU(), Nx, Ny, Nz, z_faces)

bottom_height = data1init["bottom_height"]
T_init        = data1init["T_init"]
e_init        = data1init["e_init"]
wind_stress   = data1init["u_wind_stress"]

landmask = bottom_height .> -1e-4

#dkappaT_init = data1init["dkappaT_init"]
#dkappaS_init = data1init["dkappaS_init"]

data1final = jldopen(graph_directory1 * "data_final.jld2", "r")

T_final = data1final["T_final"]
S_final = data1final["S_final"]
e_final = data1final["e_final"]
ssh     = data1final["ssh"]

u = data1final["u"]
v = data1final["v"]
w = data1final["w"]

zonal_transport = data1final["zonal_transport"]

@show zonal_transport

#
# Then we set up the node points:
#

xζ, yζ, zζ = nodes(grid, Face(), Face(), Center())
xc, yc, zc = nodes(grid, Center(), Center(), Center())
xw, yw, zw = nodes(grid, Center(), Center(), Face())


xu, yu, zu = nodes(grid, Face(), Center(), Center())
xv, yv, zv = nodes(grid, Center(), Face(), Center())

j′ = round(Int, grid.Ny / 2)
y′ = yζ[j′]


@show grid
@show zc
@show zw

du_wind_stress = data1final["du_wind_stress"]
dv_wind_stress = data1final["dv_wind_stress"]
dT             = data1final["dT"]
dS             = data1final["dS"]
dT_flux        = data1final["dT_flux"]
dkappaT_final  = data1final["dkappaT_final"]
dkappaS_final  = data1final["dkappaS_final"]
close(data1final)

using GLMakie

function plot_variables_four_panels_2x2(
    p1_xy, p2_xy, p3_xy, p4_xy,
    x1, y1, x2, y2, x3, y3, x4, y4,
    p1_title, p2_title, p3_title, p4_title,
    p1_cbt, p2_cbt, p3_cbt, p4_cbt,
    filename, landmask;
    p1_min=false, p2_min=false, p3_min=false, p4_min=false,
    p1_set_min=nothing, p2_set_min=nothing, p3_set_min=nothing, p4_set_min=nothing,
    p1_set_max=nothing, p2_set_max=nothing, p3_set_max=nothing, p4_set_max=nothing,
    shared_left=false, shared_right=false,
    color1=:balance, color2=:balance, color3=:balance, color4=:balance)

    # Compute mins and maxes
    pmins = [2 * min(0, minimum(p)) - 1e-12 for p in (p1_xy,p2_xy,p3_xy,p4_xy)]
    pmaxs = [2 * max(0, maximum(p)) + 1e-12 for p in (p1_xy,p2_xy,p3_xy,p4_xy)]

    @show pmins
    @show pmaxs

    flags = (p1_min,p2_min,p3_min,p4_min)
    for i in 1:4
        if !flags[i]
            mmax = max(pmaxs[i], -pmins[i])
            pmaxs[i] = mmax
            pmins[i] = -mmax
        end
    end

    p_set_min = [p1_set_min, p2_set_min, p3_set_min, p4_set_min]
    for i in 1:4
        if p_set_min[i] != nothing
            pmins[i] = p_set_min[i]
        end
    end

    p_set_max = [p1_set_max, p2_set_max, p3_set_max, p4_set_max]
    for i in 1:4
        if p_set_max[i] != nothing
            pmaxs[i] = p_set_max[i]
        end
    end

    # Column-wise shared symmetric limits
    left_inds  = (1,2, 3)
    right_inds = (2,4)

    if shared_left
        mmaxL = maximum(abs.(vcat(pmaxs[1], pmaxs[2], pmaxs[3])))
        mminL = minimum(vcat(pmins[1], pmins[2], pmins[3]))
        for i in left_inds
            pmins[i] = mminL
            pmaxs[i] = mmaxL
        end
    end

    if shared_right
        mmaxR = maximum(abs.(vcat(pmaxs[2], pmaxs[4])))
        mminR = minimum(vcat(pmins[2], pmins[4]))
        for i in right_inds
            pmins[i] = mminR
            pmaxs[i] = mmaxR
        end
    end

    # Limits & contour levels
    plims   = [(pmins[i], pmaxs[i]) .* 0.8 for i in 1:4]
    levels  = [[range(plims[i][1], plims[i][2], length=60)...] for i in 1:4]

    # scaled coords
    xs = [x1,x2,x3,x4] .|> x -> x .* 1e-3
    ys = [y1,y2,y3,y4] .|> y -> y .* 1e-3
    ps = [p1_xy,p2_xy,p3_xy,p4_xy]
    titles = [p1_title, p2_title, p3_title, p4_title]
    cb_titles = [p1_cbt, p2_cbt, p3_cbt, p4_cbt]
    colors = [color1, color2, color3, color4]

    fig = Figure(resolution=(1500,1800))

    idx = 1
    for r in 1:2
        for c in 1:2
            ax = Axis(fig[r, 2c-1],
                title = titles[idx],
                xlabel = (idx in (3,4) ? "x (km)" : ""),
                ylabel = (c == 1 ? "y (km)" : ""),
                aspect = 0.75,
                titlesize = 28,
                xlabelsize = 24,
                ylabelsize = 24,
                xticklabelsize = 24,
                yticklabelsize = 24,
                leftspinevisible = false,
                rightspinevisible = false,
                topspinevisible = false,
                bottomspinevisible = false)

            # smooth raster shading + contour overlay
            masked_field = copy(ps[idx])
            cont = contourf!(ax, xs[idx], ys[idx], masked_field,
                    colormap=colors[idx],
                    nan_color = :gray50,
                    levels=levels[idx],
                    colorscale=plims[idx])

            # Panel colorbar
            Colorbar(fig[r, 2c];
                colormap = to_colormap(colors[idx]),
                limits = plims[idx],
                label = cb_titles[idx],
                labelsize = 24,
                ticklabelsize = 20,
            )

            idx += 1
        end
    end

    save(filename, fig)
    return fig
end

function plot_variables_two_panels_1x2(
    p1_xy, p2_xy,
    x1, y1, x2, y2,
    p1_title, p2_title,
    p1_cbt, p2_cbt,
    filename, landmasks;
    p1_min=false, p2_min=false,
    p1_set_min=nothing, p2_set_min=nothing,
    p1_set_max=nothing, p2_set_max=nothing,
    shared=false,
    color1=:balance, color2=:balance)

   # Compute mins and maxes
    pmins = [2 * min(0, minimum(p)) - 1e-12 for p in (p1_xy,p2_xy)]
    pmaxs = [2 * max(0, maximum(p)) + 1e-12 for p in (p1_xy,p2_xy)]

    @show pmins
    @show pmaxs

    flags = (p1_min,p2_min)
    for i in 1:2
        if !flags[i]
            mmax = max(pmaxs[i], -pmins[i])
            pmaxs[i] = mmax
            pmins[i] = -mmax
        end
    end

    p_set_min = [p1_set_min, p2_set_min]
    for i in 1:2
        if p_set_min[i] != nothing
            pmins[i] = p_set_min[i]
        end
    end

    p_set_max = [p1_set_max, p2_set_max]
    for i in 1:2
        if p_set_max[i] != nothing
            pmaxs[i] = p_set_max[i]
        end
    end

    if shared
        mmaxL = maximum(abs.(vcat(pmaxs[1], pmaxs[2])))
        mminL = minimum(vcat(pmins[1], pmins[2]))
        for i in 1:2
            pmins[i] = mminL
            pmaxs[i] = mmaxL
        end
    end

    # Limits & contour levels
    plims   = [(pmins[i], pmaxs[i]) .* 0.8 for i in 1:2]
    levels  = [[range(plims[i][1], plims[i][2], length=60)...] for i in 1:2]

    # scaled coords
    xs = [x1,x2] .|> x -> x .* 1e-3
    ys = [y1,y2] .|> y -> y .* 1e-3
    ps = [p1_xy,p2_xy]
    titles = [p1_title, p2_title]
    cb_titles = [p1_cbt, p2_cbt]
    colors = [color1, color2]

    fig = Figure(resolution=(1500,1000))

    idx = 1
    for r in 1:1
        for c in 1:2
            ax = Axis(fig[r, 2c-1],
                title = titles[idx],
                xlabel = (idx in (3,4) ? "x (km)" : ""),
                ylabel = (c == 1 ? "y (km)" : ""),
                aspect = 0.75,
                titlesize = 28,
                xlabelsize = 24,
                ylabelsize = 24,
                xticklabelsize = 24,
                yticklabelsize = 24,
                leftspinevisible = false,
                rightspinevisible = false,
                topspinevisible = false,
                bottomspinevisible = false)

            # smooth raster shading + contour overlay
            masked_field = copy(ps[idx])
            masked_field[landmasks[idx]] .= NaN
            cont = contourf!(ax, xs[idx], ys[idx], masked_field,
                    colormap=colors[idx],
                    nan_color = :gray50,
                    levels=levels[idx],
                    colorscale=plims[idx])

            # Panel colorbar
            Colorbar(fig[r, 2c];
                colormap = to_colormap(colors[idx]),
                limits = plims[idx],
                label = cb_titles[idx],
                labelsize = 24,
                ticklabelsize = 20
            )

            idx += 1
        end
    end

    save(filename, fig)
    return fig
end

function plot_variables_two_panels_2x1(
    p1_xy, p2_xy,
    x1, y1, x2, y2,
    p1_title, p2_title,
    p1_cbt, p2_cbt,
    filename;
    p1_min=false, p2_min=false,
    p1_set_min=nothing, p2_set_min=nothing,
    p1_set_max=nothing, p2_set_max=nothing,
    shared=false,
    color1=:balance, color2=:balance)

    # Compute mins and maxes
    pmins = [2 * min(0, minimum(p)) - 1e-12 for p in (p1_xy,p2_xy)]
    pmaxs = [2 * max(0, maximum(p)) + 1e-12 for p in (p1_xy,p2_xy)]

    @show pmins
    @show pmaxs

    flags = (p1_min,p2_min)
    for i in 1:2
        if !flags[i]
            mmax = max(pmaxs[i], -pmins[i])
            pmaxs[i] = mmax
            pmins[i] = -mmax
        end
    end

    p_set_min = [p1_set_min, p2_set_min]
    for i in 1:2
        if p_set_min[i] != nothing
            pmins[i] = p_set_min[i]
        end
    end

    p_set_max = [p1_set_max, p2_set_max]
    for i in 1:2
        if p_set_max[i] != nothing
            pmaxs[i] = p_set_max[i]
        end
    end

    if shared
        mmaxL = maximum(abs.(vcat(pmaxs[1], pmaxs[2])))
        mminL = minimum(vcat(pmins[1], pmins[2]))
        for i in 1:2
            pmins[i] = mminL
            pmaxs[i] = mmaxL
        end
    end

    # Limits & contour levels
    plims   = [(pmins[i], pmaxs[i]) .* 0.8 for i in 1:2]
    levels  = [[range(plims[i][1], plims[i][2], length=60)...] for i in 1:2]

    # scaled coords
    xs = [x1,x2] .|> x -> x .* 1e-3
    ys = [y1,y2] .|> y -> y .* 1e-3
    ps = [p1_xy,p2_xy]
    titles = [p1_title, p2_title]
    cb_titles = [p1_cbt, p2_cbt]
    colors = [color1, color2]

    fig = Figure(resolution=(2000,1200))

    idx = 1
    for r in 1:2
        c = 1
        ax = Axis(fig[r, 2c-1],
            title = titles[idx],
            xlabel = (idx in (1,2) ? "x (km)" : ""),
            ylabel = (c == 1 ? "y (km)" : ""),
            aspect = 4,
            titlesize = 28,
            xlabelsize = 24,
            ylabelsize = 24,
            xticklabelsize = 24,
            yticklabelsize = 24,
            leftspinevisible = false,
            rightspinevisible = false,
            topspinevisible = false,
            bottomspinevisible = false)

        # smooth raster shading + contour overlay
        masked_field = copy(ps[idx])
        cont = contourf!(ax, xs[idx], ys[idx], masked_field,
                    colormap=colors[idx],
                    nan_color = :gray50,
                    levels=levels[idx],
                    colorscale=plims[idx])

        # Panel colorbar
        Colorbar(fig[r, 2c];
            colormap = to_colormap(colors[idx]),
            limits = plims[idx],
            label = cb_titles[idx],
            labelsize = 24,
            ticklabelsize = 20,
        )

        idx += 1
    end

    save(filename, fig)
    return fig
end


landmask_center = landmask
landmask_u = landmask_center #landmask_center[1:end-1, :] .|| landmask_center[2:end, :]

landmask_v = falses(Nx, Ny+1)
landmask_v[:, 2:Ny] = landmask_center[:, 1:Ny-1] .| landmask_center[:, 2:Ny]
landmask_v[:, 1] = landmask_center[:, 1]
landmask_v[:, Ny+1] = landmask_center[:, Ny]

@show size(landmask_u)
@show size(landmask_v)
@show size(dv_wind_stress)

landmask_gradients = [landmask_center, landmask_v, landmask_center, landmask_center, landmask_center, landmask_center]
landmask_velocities = [landmask_u, landmask_v, landmask_u, landmask_v, landmask_u, landmask_v]
landmask_centers = [landmask_center, landmask_center, landmask_center, landmask_center, landmask_center, landmask_center]

stacked_landmask = repeat(landmask, 1, 1, Nz)
@show minimum(S_final[.!stacked_landmask]), maximum(S_final[.!stacked_landmask])

plot_variables_four_panels_2x2(dT[:,:,31], dT[:,:,14], dkappaT_final[:,:,31], dkappaT_final[:,:,14], xc, yc, xc, yc, xc, yc, xc, yc,
                          "(a) ∂J/∂T(x, y, 15m)", "(b) ∂J/∂T(x, y, 504m)", "(c) ∂J/∂κₜ(x, y, 15m)", "(d) ∂J/∂κₜ(x, y, 504m)",
                          "Sv / °C", "Sv / °C", "Sv / m²s⁻¹", "Sv / m²s⁻¹",
                          graph_directory1 * "gradients_Tdiff_xy.png", landmask)

plot_variables_four_panels_2x2(T_final[:,:,31], T_final[:,:,14], T_final[:,:,4], ssh[:,:], xc, yc, xc, yc, xc, yc, xc, yc,
                          "(a) T(x, y, z=15m)", "(b) T(x, y, z=504m)", "(c) T(x, y, z=1518m)", "(d) SSH(x, y)",
                          "°C", "°C", "°C", "m",
                          graph_directory1 * "tracers_oceananigans_xy.png", landmask;
                          color1=:thermal, color2=:thermal, color3=:thermal,
                          p1_min=true, p2_min=true, p3_min=true, p4_min=true,
                          #p1_set_max=10.0, p2_set_max=8.0, p3_set_max=3.0, p4_set_max=2.0,
                          p1_set_min=-1.0, p2_set_min=-1.0, p3_set_min=-1.0)

plot_variables_four_panels_2x2(u[:,:,31], u[:,:,14], v[:,:,31], v[:,:,14], xu, yu, xu, yu, xv, yv, xv, yv,
                          "(a) u(x, y, 15m)", "(b) u(x, y, 504m)", "(c) v(x, y, 15m)", "(d) v(x, y, 504m)",
                          "m / s", "m / s", "m / s", "m / s",
                          graph_directory1 * "velocities_xy.png", landmask)


j′ = round(Int, grid.Ny / 2)

landmask_gradients = [landmask_center, landmask_v]
plot_variables_two_panels_1x2(du_wind_stress[:,:,1], dv_wind_stress[:,:,1], xu, yu, xv, yv,
                          "(a) ∂J/∂τₓ(x, y)", "(b) ∂J/∂τᵧ(x, y)",
                          "Sv / m²s⁻²", "Sv / m²s⁻²",
                          graph_directory1 * "gradients_windstress_xy.png", landmask_gradients)
                          #p1_set_min=-600, p2_set_min=-600, p1_set_max=600, p2_set_max=600)

z_thicknesses = zw[2:end] - zw[1:end-1]

plot_variables_two_panels_2x1(dT[44, :, :] ./ z_thicknesses', dT[:,j′,:] ./ z_thicknesses', yc, zc, xc, zc,
                          "(a) ∂J/∂T(x=550km, y, z)", "(b) ∂J/∂T(x, y=1000km, z)",
                          "Sv / °C", "Sv / °C",
                          graph_directory1 * "gradients_oceananigans_depth.png")
                          #p1_set_min=-7e-5, p2_set_min=-7e-5, p1_set_max=7e-5, p2_set_max=7e-5)

plot_variables_two_panels_2x1(T_final[44, :, :], T_final[:,j′,:], yc, zc, xc, zc,
                          "(a) T(x=550km, y, z)", "(b) T(x, y=1000km, z)",
                          "°C", "°C",
                          graph_directory1 * "temperature_depth.png";
                          color1=:thermal, color2=:thermal,
                          p1_min=true, p2_min=true)
                          #p1_set_min=-1.0, p2_set_min=-1.0, p1_set_max=10.0, p2_set_max=6.0)

plot_variables_two_panels_2x1(u[44, :, :], u[:,j′,:], yu, zu, xu, zu,
                          "(a) u(x=550km, y, z)", "(b) u(x, y=1000km, z)",
                          "ms⁻¹", "ms⁻¹",
                          graph_directory1 * "u_depth.png")
                          #p1_set_min=-1.3, p2_set_min=-1.3, p1_set_max=1.3, p2_set_max=1.3)

plot_variables_two_panels_2x1(v[44, :, :], v[:,j′,:], yv, zv, xv, zv,
                          "(a) v(x=550km, y, z)", "(b) v(x, y=1000km, z)",
                          "ms⁻¹", "ms⁻¹",
                          graph_directory1 * "v_depth.png")
                          #p1_set_min=-1.0, p2_set_min=-1.0, p1_set_max=1.0, p2_set_max=1.0)

data2final = jldopen(graph_directory2 * "data_final.jld2", "r")

T_final2 = data2final["T_final"]
S_final2 = data2final["S_final"]
e_final2 = data2final["e_final"]
ssh2     = data2final["ssh"]

u2 = data2final["u"]
v2 = data2final["v"]
w2 = data2final["w"]

zonal_transport = data2final["zonal_transport"]

@show zonal_transport

#
# Then we set up the node points:
#

xζ, yζ, zζ = nodes(grid, Face(), Face(), Center())
xc, yc, zc = nodes(grid, Center(), Center(), Center())
xw, yw, zw = nodes(grid, Center(), Center(), Face())


xu, yu, zu = nodes(grid, Face(), Center(), Center())
xv, yv, zv = nodes(grid, Center(), Face(), Center())

j′ = round(Int, grid.Ny / 2)
y′ = yζ[j′]


@show grid
@show zc
@show zw

du_wind_stress2 = data2final["du_wind_stress"]
dv_wind_stress2 = data2final["dv_wind_stress"]
dT2             = data2final["dT"]
dS2             = data2final["dS"]
dT_flux2        = data2final["dT_flux"]
dkappaT_final2  = data2final["dkappaT_final"]
dkappaS_final2  = data2final["dkappaS_final"]
close(data2final)

dT_error = abs.(dT2 .- dT) ./ abs.(dT)
dkappaT_final_error = abs.(dkappaT_final2 .- dkappaT_final) ./ abs.(dkappaT_final)

plot_variables_four_panels_2x2(dT_error[:,:,31], dT_error[:,:,14], dkappaT_final_error[:,:,31], dkappaT_final_error[:,:,14], xc, yc, xc, yc, xc, yc, xc, yc,
                          "(a) Error ∂J/∂T(x, y, 15m)", "(b) Error ∂J/∂T(x, y, 504m)", "(c) Error ∂J/∂κₜ(x, y, 15m)", "(d) Error ∂J/∂κₜ(x, y, 504m)",
                          "Sv / °C", "Sv / °C", "Sv / m²s⁻¹", "Sv / m²s⁻¹",
                          graph_directory2 * "error_gradients_Tdiff_xy.png", landmask)


u_error = abs.(u2 .- u) ./ abs.(u .+ 1)
v_error = abs.(v2 .- v) ./ abs.(v .+ 1)

plot_variables_four_panels_2x2(u_error[:,:,31], u_error[:,:,14], v_error[:,:,31], v_error[:,:,14], xu, yu, xu, yu, xv, yv, xv, yv,
                          "(a) Error u(x, y, 15m)", "(b) Error u(x, y, 504m)", "(c) Error v(x, y, 15m)", "(d) Error v(x, y, 504m)",
                          "m / s", "m / s", "m / s", "m / s",
                          graph_directory2 * "error_velocities_xy.png", landmask)


T_final_error = abs.(T_final2 .- T_final) ./ abs.(T_final .+ 1)
ssh_error = abs.(ssh2 .- ssh) ./ abs.(ssh .+ 1)

plot_variables_four_panels_2x2(T_final_error[:,:,31], T_final_error[:,:,14], T_final_error[:,:,4], ssh_error[:,:], xc, yc, xc, yc, xc, yc, xc, yc,
                          "(a) Error T(x, y, z=15m)", "(b) Error T(x, y, z=504m)", "(c) Error T(x, y, z=1518m)", "(d) Error SSH(x, y)",
                          "°C", "°C", "°C", "m",
                          graph_directory2 * "error_tracers_oceananigans_xy.png", landmask;
                          color1=:thermal, color2=:thermal, color3=:thermal)

plot_variables_two_panels_1x2(du_wind_stress[:,:,1], dv_wind_stress[:,:,1], xu, yu, xv, yv,
                          "(a) J/∂τₓ(x, y)", "(b) ∂J/∂τᵧ(x, y)",
                          "Sv / m²s⁻²", "Sv / m²s⁻²",
                          graph_directory1 * "gradients_windstress_xy.png", landmask_gradients)



plot_variables_two_panels_2x1(dT_error[44, :, :] ./ z_thicknesses', dT_error[:,j′,:] ./ z_thicknesses', yc, zc, xc, zc,
                          "(a) Error ∂J/∂T(x=550km, y, z)", "(b) Error ∂J/∂T(x, y=1000km, z)",
                          "Sv / °C", "Sv / °C",
                          graph_directory2 * "error_gradients_oceananigans_depth.png")

plot_variables_two_panels_2x1(T_final_error[44, :, :], T_final_error[:,j′,:], yc, zc, xc, zc,
                          "(a) Error T(x=550km, y, z)", "(b) Error T(x, y=1000km, z)",
                          "°C", "°C",
                          graph_directory2 * "error_temperature_depth.png";
                          color1=:thermal, color2=:thermal,
                          p1_min=true, p2_min=true)

plot_variables_two_panels_2x1(u_error[44, :, :], u_error[:,j′,:], yu, zu, xu, zu,
                          "(a) Error u(x=550km, y, z)", "(b) Error u(x, y=1000km, z)",
                          "ms⁻¹", "ms⁻¹",
                          graph_directory2 * "error_u_depth.png")

plot_variables_two_panels_2x1(v_error[44, :, :], v_error[:,j′,:], yv, zv, xv, zv,
                          "(a) Error v(x=550km, y, z)", "(b) Error v(x, y=1000km, z)",
                          "ms⁻¹", "ms⁻¹",
                          graph_directory2 * "error_v_depth.png")


du_wind_stress_error = abs.(du_wind_stress2 .- du_wind_stress) ./ abs.(du_wind_stress .+ 1)
dv_wind_stress_error = abs.(dv_wind_stress2 .- dv_wind_stress) ./ abs.(dv_wind_stress .+ 1)

plot_variables_two_panels_1x2(du_wind_stress_error[:,:,1], dv_wind_stress_error[:,:,1], xu, yu, xv, yv,
                          "(a) Error ∂J/∂τₓ(x, y)", "(b) Error ∂J/∂τᵧ(x, y)",
                          "Sv / m²s⁻²", "Sv / m²s⁻²",
                          graph_directory2 * "error_gradients_windstress_xy.png", landmask_gradients)