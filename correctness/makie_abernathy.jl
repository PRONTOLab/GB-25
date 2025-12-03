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

graph_directory = "run_abernathy_model_ad_spinup4000000_8100steps_noImmersedGrid/"
#graph_directory = "run_abernathy_model_ad_spinup4000000_8100steps_zeroImmersedGrid/"

#
# First we gather the data and create a grid for plotting purposes:
#
data1 = jldopen(graph_directory * "data_init.jld2", "r")

Nx = data1["Nx"]
Ny = data1["Ny"]
Nz = data1["Nz"]

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

depth = grid.immersed_boundary.bottom_height[1:Nx,1:Ny,1]
landmask = depth .> -1e-4

bottom_height = data1["bottom_height"]
T_init        = data1["T_init"]
e_init        = data1["e_init"]
wind_stress   = data1["u_wind_stress"]

#dkappaT_init = data1["dkappaT_init"]
#dkappaS_init = data1["dkappaS_init"]

data2 = jldopen(graph_directory * "data_final.jld2", "r")

T_final = data2["T_final"]
S_final = data2["S_final"]
e_final = data2["e_final"]
ssh     = data2["ssh"]

u = data2["u"]
v = data2["v"]
w = data2["w"]

zonal_transport = data2["zonal_transport"]

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

du_wind_stress = data2["du_wind_stress"]
dv_wind_stress = data2["dv_wind_stress"]
dT             = data2["dT"]
dS             = data2["dS"]
dT_flux        = data2["dT_flux"]
dkappaT_final  = data2["dkappaT_final"]
dkappaS_final  = data2["dkappaS_final"]
close(data2)

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
    pmins = [minimum(p[.!landmask]) - 1e-12 for p in (p1_xy,p2_xy,p3_xy,p4_xy)]
    pmaxs = [maximum(p[.!landmask]) + 1e-12 for p in (p1_xy,p2_xy,p3_xy,p4_xy)]

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
            masked_field[landmask] .= NaN
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
    pmins = [minimum(p) for p in (p1_xy,p2_xy)]
    pmaxs = [maximum(p) for p in (p1_xy,p2_xy)]

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
    pmins = [minimum(p) for p in (p1_xy,p2_xy)]
    pmaxs = [maximum(p) for p in (p1_xy,p2_xy)]

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
                          graph_directory * "gradients_Tdiff_xy.png", landmask)

plot_variables_four_panels_2x2(T_final[:,:,31], T_final[:,:,14], T_final[:,:,4], ssh[:,:], xc, yc, xc, yc, xc, yc, xc, yc,
                          "(a) T(x, y, z=15m)", "(b) T(x, y, z=504m)", "(c) T(x, y, z=1518m)", "(d) SSH(x, y)",
                          "°C", "°C", "°C", "m",
                          graph_directory * "tracers_oceananigans_xy.png", landmask;
                          color1=:thermal, color2=:thermal, color3=:thermal,
                          p1_min=true, p2_min=true, p3_min=true, p4_min=true,
                          p1_set_max=10.0, p2_set_max=8.0, p3_set_max=3.0, p4_set_max=2.0,
                          p1_set_min=-1.0, p2_set_min=-1.0, p3_set_min=-1.0, p4_set_min=-2.0)


j′ = round(Int, grid.Ny / 2)

landmask_gradients = [landmask_center, landmask_v]
plot_variables_two_panels_1x2(du_wind_stress[:,:,1], dv_wind_stress[:,:,1], xu, yu, xv, yv,
                          "(a) ∂J/∂τₓ(x, y)", "(b) ∂J/∂τᵧ(x, y)",
                          "Sv / m²s⁻²", "Sv / m²s⁻²",
                          graph_directory * "gradients_windstress_xy.png", landmask_gradients;
                          p1_set_min=-600, p2_set_min=-600, p1_set_max=600, p2_set_max=600)

z_thicknesses = zw[2:end] - zw[1:end-1]

plot_variables_two_panels_2x1(dT[44, :, :] ./ z_thicknesses', dT[:,j′,:] ./ z_thicknesses', yc, zc, xc, zc,
                          "(a) ∂J/∂T(x=550km, y, z)", "(b) ∂J/∂T(x, y=1000km, z)",
                          "Sv / °C", "Sv / °C",
                          graph_directory * "gradients_oceananigans_depth.png";
                          p1_set_min=-7e-5, p2_set_min=-7e-5, p1_set_max=7e-5, p2_set_max=7e-5)

plot_variables_two_panels_2x1(T_final[44, :, :], T_final[:,j′,:], yc, zc, xc, zc,
                          "(a) T(x=550km, y, z)", "(b) T(x, y=1000km, z)",
                          "°C", "°C",
                          graph_directory * "temperature_depth.png";
                          color1=:thermal, color2=:thermal,
                          p1_min=true, p2_min=true,
                          p1_set_min=-1.0, p2_set_min=-1.0, p1_set_max=10.0, p2_set_max=6.0)

plot_variables_two_panels_2x1(u[44, :, :], u[:,j′,:], yu, zu, xu, zu,
                          "(a) u(x=550km, y, z)", "(b) u(x, y=1000km, z)",
                          "ms⁻¹", "ms⁻¹",
                          graph_directory * "u_depth.png";
                          p1_set_min=-1.3, p2_set_min=-1.3, p1_set_max=1.3, p2_set_max=1.3)

plot_variables_two_panels_2x1(v[44, :, :], v[:,j′,:], yv, zv, xv, zv,
                          "(a) v(x=550km, y, z)", "(b) v(x, y=1000km, z)",
                          "ms⁻¹", "ms⁻¹",
                          graph_directory * "v_depth.png";
                          p1_set_min=-1.0, p2_set_min=-1.0, p1_set_max=1.0, p2_set_max=1.0)