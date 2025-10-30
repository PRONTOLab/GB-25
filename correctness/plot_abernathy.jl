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

using Plots
#pyplot()
gr(
    left_margin = 12Plots.mm,
    top_margin = 4Plots.mm,
)

# Force scientific tick labels globally
default(colorbar_formatter = :scientific)

graph_directory = "run_abernathy_model_ad_spinup5000_8100steps_KindaHiRes_linearEOS_noCATKE_halfTimeStep_highVisc_WENOOrder3_gridFittedBottom_wallRidge_scaledVerticalDiff/"

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
    smoothed_ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, wall_function)

    grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(smoothed_ridge))
    return grid
end

grid = make_grid(CPU(), Nx, Ny, Nz, z_faces)

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

#
# First a plot of velocities:
#

# Also get plots of everything at indices 14 (depth 533.9 m) and 25 (depth 116m)

function plot_variables(p1_xz, p2_xy, p3_xy,
                        x1, z1, x2, y2, x3, y3,
                        p1_title, p2_title, p3_title,
                        filename;
                        p1_min=false, p2_min=false, p3_min=false, shared=false, color=:balance)

    wmax = maximum(p1_xz)
    umax = maximum(p2_xy)
    vmax = maximum(p3_xy)

    wmin = minimum(p1_xz)
    umin = minimum(p2_xy)
    vmin = minimum(p3_xy)

    if !p1_min
        wmax = max(wmax, -wmin)
        wmin = -wmax
    end

    if !p2_min
        umax = max(umax, -umin)
        umin = -umax
    end

    if !p3_min
        vmax = max(vmax, -vmin)
        vmin = -vmax
    end

    if shared
        umax = max(umax, vmax)
        umin = min(umin, vmin)
        vmax = umax
        vmin = umin
    end

    wlims = (wmin, wmax) .* 0.8
    ulims = (umin, umax) .* 0.8
    vlims = (vmin, vmax) .* 0.8

    wlevels = vcat([wmin], range(wlims[1], wlims[2], length = 31), [wmax])
    ulevels = vcat([umin], range(ulims[1], ulims[2], length = 31), [umax])
    vlevels = vcat([vmin], range(vlims[1], vlims[2], length = 31), [vmax])

    xlims = (0, grid.Lx) .* 1e-3
    ylims = (0, grid.Ly) .* 1e-3
    zlims = (-grid.Lz, 0)

    p1_xz_plot = contourf(x1 * 1e-3, z1, p1_xz',
        xlabel = "y (km)",
        ylabel = "z (m)",
        aspectratio = 0.05,
        linewidth = 0,
        levels = wlevels,
        clims = wlims,
        xlims = ylims,
        ylims = zlims,
        color = color)

    p2_xy_plot = contourf(x2 * 1e-3, y2 * 1e-3, p2_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = ulevels,
        clims = ulims,
        xlims = xlims,
        ylims = ylims,
        color = color)

    p3_xy_plot = contourf(x3 * 1e-3, y3 * 1e-3, p3_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = vlevels,
        clims = vlims,
        xlims = xlims,
        ylims = ylims,
        color = color)

    layout = @layout [upper_slice_plot{0.2h}
        Plots.grid(1, 2)]

    plot(p1_xz_plot, p2_xy_plot, p3_xy_plot, layout = layout, size = (1200, 1200), title = [p1_title p2_title p3_title])

    savefig(filename)
end

function plot_variables_two_panels_updown(p1_xy, p2_xy,
                        x1, y1, x2, y2,
                        p1_title, p2_title,
                        p1_colorbar_title, p2_colorbar_title,
                        filename;
                        p1_min=false, p2_min=false, color=:balance)

    p1max = maximum(p1_xy)
    p2max = maximum(p2_xy)

    p1min = minimum(p1_xy)
    p2min = minimum(p2_xy)

    if !p1_min
        p1max = max(p1max, -p1min)
        p1min = -p1max
    end
    if !p2_min
        p2max = max(p2max, -p2min)
        p2min = -p2max
    end

    p1lims = (p1min, p1max) .* 0.8
    p2lims = (p2min, p2max) .* 0.8

    p1levels = vcat([p1min], range(p1lims[1], p1lims[2], length = 62), [p1max])
    p2levels = vcat([p2min], range(p2lims[1], p2lims[2], length = 62), [p2max])

    xlims = (0, grid.Lx) .* 1e-3
    ylims = (0, grid.Ly) .* 1e-3
    zlims = (-grid.Lz, 0)

    p1_xy_plot = contourf(x1 * 1e-3, y1, p1_xy',
        xlabel = "y (km)",
        ylabel = "z (m)",
        aspectratio = 0.2,
        linewidth = 0,
        levels = p1levels,
        clims = p1lims,
        colorbar_title = p1_colorbar_title,
        xlims = ylims,
        ylims = zlims,
        tickfontsize = 12,
        color = color,
        bottom_margin = 16Plots.mm,
        )

    p2_xy_plot = contourf(x2 * 1e-3, y2, p2_xy',
        xlabel = "x (km)",
        ylabel = "z (m)",
        aspectratio = 0.1,
        linewidth = 0,
        levels = p2levels,
        clims = p2lims,
        colorbar_title = p2_colorbar_title,
        xlims = xlims,
        ylims = zlims,
        tickfontsize = 12,
        color = color,
        )

    #annotate!(p1_xy_plot, (0.02, 1.80, text("(a)", :left, 14, :black)))
    #annotate!(p2_xy_plot, (0.02, 1.80, text("(b)", :left, 14, :black)))

    layout = @layout [Plots.grid(1, 1)
                      Plots.grid(1, 1)]

    p = plot(p1_xy_plot, p2_xy_plot, layout = layout, size = (1800, 1000), title = [p1_title p2_title])

    savefig(filename)
end

function plot_variables_four_panels(p1_xy, p2_xy, p3_xy, p4_xy,
                        x1, y1, x2, y2, x3, y3, x4, y4,
                        p1_title, p2_title, p3_title, p4_title,
                        filename;
                        p1_min=false, p2_min=false, p3_min=false, p4_min=false, color=:balance)

    p1max = maximum(p1_xy)
    p2max = maximum(p2_xy)
    p3max = maximum(p3_xy)
    p4max = maximum(p4_xy)

    p1min = minimum(p1_xy)
    p2min = minimum(p2_xy)
    p3min = minimum(p3_xy)
    p4min = minimum(p4_xy)

    if !p1_min
        p1max = max(p1max, -p1min)
        p1min = -p1max
    end

    if !p2_min
        p2max = max(p2max, -p2min)
        p2min = -p2max
    end

    if !p3_min
        p3max = max(p3max, -p3min)
        p3min = -p3max
    end

    if !p4_min
        p4max = max(p4max, -p4min)
        p4min = -p4max
    end

    p1lims = (p1min, p1max) .* 0.8
    p2lims = (p2min, p2max) .* 0.8
    p3lims = (p3min, p3max) .* 0.8
    p4lims = (p4min, p4max) .* 0.8

    p1levels = vcat([p1min], range(p1lims[1], p1lims[2], length = 31), [p1max])
    p2levels = vcat([p2min], range(p2lims[1], p2lims[2], length = 31), [p2max])
    p3levels = vcat([p3min], range(p3lims[1], p3lims[2], length = 31), [p3max])
    p4levels = vcat([p4min], range(p4lims[1], p4lims[2], length = 31), [p4max])

    xlims = (0, grid.Lx) .* 1e-3
    ylims = (0, grid.Ly) .* 1e-3
    zlims = (-grid.Lz, 0)

    p1_xy_plot = contourf(x1 * 1e-3, y1 * 1e-3, p1_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p1levels,
        clims = p1lims,
        xlims = xlims,
        ylims = ylims,
        color = color)

    p2_xy_plot = contourf(x2 * 1e-3, y2 * 1e-3, p2_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p2levels,
        clims = p2lims,
        xlims = xlims,
        ylims = ylims,
        color = color)

    p3_xy_plot = contourf(x3 * 1e-3, y3 * 1e-3, p3_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p3levels,
        clims = p3lims,
        xlims = xlims,
        ylims = ylims,
        color = color)

    p4_xy_plot = contourf(x4 * 1e-3, y4 * 1e-3, p4_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p4levels,
        clims = p4lims,
        xlims = xlims,
        ylims = ylims,
        color = color)

    layout = @layout [Plots.grid(1, 2)
                      Plots.grid(1, 2)]

    plot(p1_xy_plot, p2_xy_plot, p3_xy_plot, p4_xy_plot, layout = layout, size = (1200, 1200), title = [p1_title p2_title p3_title p4_title])

    savefig(filename)
end

function plot_variables_six_panels(p1_xy, p2_xy, p3_xy, p4_xy, p5_xy, p6_xy,
                        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6,
                        p1_title, p2_title, p3_title, p4_title, p5_title, p6_title,
                        p1_colorbar_title, p2_colorbar_title, p3_colorbar_title, p4_colorbar_title, p5_colorbar_title, p6_colorbar_title,
                        filename;
                        p1_min=false, p2_min=false, p3_min=false, p4_min=false, p5_min=false, p6_min=false, shared=false, color=:balance)

    p1max = maximum(p1_xy)
    p2max = maximum(p2_xy)
    p3max = maximum(p3_xy)
    p4max = maximum(p4_xy)
    p5max = maximum(p5_xy)
    p6max = maximum(p6_xy)

    p1min = minimum(p1_xy)
    p2min = minimum(p2_xy)
    p3min = minimum(p3_xy)
    p4min = minimum(p4_xy)
    p5min = minimum(p5_xy)
    p6min = minimum(p6_xy)

    if !p1_min
        p1max = max(p1max, -p1min)
        p1min = -p1max
    end
    if !p2_min
        p2max = max(p2max, -p2min)
        p2min = -p2max
    end
    if !p3_min
        p3max = max(p3max, -p3min)
        p3min = -p3max
    end
    if !p4_min
        p4max = max(p4max, -p4min)
        p4min = -p4max
    end
    if !p5_min
        p5max = max(p5max, -p5min)
        p5min = -p5max
    end
    if !p6_min
        p6max = max(p6max, -p6min)
        p6min = -p6max
    end

    if shared
        pmax = max(p1max, p2max, p3max, p4max, p5max, p6max)
        #pmin = max(p1min, p2min, p3min, p4min, p5min, p6min)
        p1max = p2max = p3max = p4max = p5max = p6max = pmax
        p1min = p2min = p3min = p4min = p5min = p6min = -pmax
    end

    p1lims = (p1min, p1max) .* 0.8
    p2lims = (p2min, p2max) .* 0.8
    p3lims = (p3min, p3max) .* 0.8
    p4lims = (p4min, p4max) .* 0.8
    p5lims = (p5min, p5max) .* 0.8
    p6lims = (p6min, p6max) .* 0.8

    p1levels = vcat([p1min], range(p1lims[1], p1lims[2], length = 62), [p1max])
    p2levels = vcat([p2min], range(p2lims[1], p2lims[2], length = 62), [p2max])
    p3levels = vcat([p3min], range(p3lims[1], p3lims[2], length = 62), [p3max])
    p4levels = vcat([p4min], range(p4lims[1], p4lims[2], length = 62), [p4max])
    p5levels = vcat([p5min], range(p5lims[1], p5lims[2], length = 62), [p5max])
    p6levels = vcat([p6min], range(p6lims[1], p6lims[2], length = 62), [p6max])

    xlims = (0, grid.Lx) .* 1e-3
    ylims = (0, grid.Ly) .* 1e-3
    zlims = (-grid.Lz, 0)

    p1_xy_plot = contourf(x1 * 1e-3, y1 * 1e-3, p1_xy',
        #xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p1levels,
        clims = p1lims,
        colorbar_title = p1_colorbar_title,
        xlims = xlims,
        ylims = ylims,
        tickfontsize = 12,
        bottom_margin = 16Plots.mm,
        color = color)

    p2_xy_plot = contourf(x2 * 1e-3, y2 * 1e-3, p2_xy',
        #xlabel = "x (km)",
        #ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p2levels,
        clims = p2lims,
        colorbar_title = p2_colorbar_title,
        xlims = xlims,
        ylims = ylims,
        tickfontsize = 12,
        bottom_margin = 16Plots.mm,
        color = color)

    p3_xy_plot = contourf(x3 * 1e-3, y3 * 1e-3, p3_xy',
        #xlabel = "x (km)",
        #ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p3levels,
        clims = p3lims,
        colorbar_title = p3_colorbar_title,
        xlims = xlims,
        ylims = ylims,
        tickfontsize = 12,
        bottom_margin = 16Plots.mm,
        color = color)

    p4_xy_plot = contourf(x4 * 1e-3, y4 * 1e-3, p4_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p4levels,
        clims = p4lims,
        colorbar_title = p4_colorbar_title,
        xlims = xlims,
        ylims = ylims,
        tickfontsize = 12,
        #colorbar_formatter = x -> @sprintf("%.2e", x),
        color = color)

    p5_xy_plot = contourf(x5 * 1e-3, y5 * 1e-3, p5_xy',
        xlabel = "x (km)",
        #ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p5levels,
        clims = p5lims,
        colorbar_title = p5_colorbar_title,
        xlims = xlims,
        ylims = ylims,
        tickfontsize = 12,
        #colorbar_formatter = x -> @sprintf("%.2e", x),
        color = color)

    p6_xy_plot = contourf(x6 * 1e-3, y6 * 1e-3, p6_xy',
        xlabel = "x (km)",
        #ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = p6levels,
        clims = p6lims,
        colorbar_title = p6_colorbar_title,
        #colorbar_ticks = ([-5e-3, -2.5e-3, 0, 2.5e-3, 5e-3], ["-5e-3", "-2.5e-3", "0", "2.5e-3", "5e-3"]), # ONLY GOOD FOR ONE PLOT
        xlims = xlims,
        ylims = ylims,
        tickfontsize = 12,
        annotationfontsize = 12,
        #colorbar_formatter = x -> @sprintf("%.2e", x),
        color = color)

    layout = @layout [Plots.grid(1, 3)
                      Plots.grid(1, 3)]

    plot(p1_xy_plot, p2_xy_plot, p3_xy_plot, p4_xy_plot, p5_xy_plot, p6_xy_plot, layout = layout, size = (1800, 1300), title = [p1_title p2_title p3_title p4_title p5_title p6_title], colorbar_formatter = :scientific)

    savefig(filename)
end

plot_variables(w[:, j′, :], u[:, :, 1], v[:, :, 1], xw, zw, xu, yu, xv, yv, "w(x, z)", "u(x, y, 2180m)", "v(x, y, 2180m)", graph_directory * "final_velocities_bottom.png")
#plot_variables(u[25, :, :], u[:, :, 14], v[:, :, 14], yu, zu, xu, yu, xv, yv, "u(y, z) (NOT x,z)", "u(x, y, 533m)", "v(x, y, 533m)", graph_directory * "final_velocities_533m.png")
plot_variables(v[:, j′, :], u[:, :, 25], v[:, :, 25], xv, zv, xu, yu, xv, yv, "v(x, z)", "u(x, y, 116m)", "v(x, y, 116m)", graph_directory * "final_velocities_116m.png")


plot_variables(T_final[:, j′, :], T_final[:, :, grid.Nz], T_final[:, :, 25], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 0m)", "T(x, y, 116m)", graph_directory * "final_T_0to116m.png")
plot_variables(T_final[:, j′, :], T_final[:, :, 16], T_final[:, :, 1], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 533m)", "T(x, y, 2180m)", graph_directory * "final_T_533to2180m.png")

#plot_variables(e_final[:, j′, :], e_final[:,:,grid.Nz], ssh[:,:,1], xc, zc, xc, yc, xc, yc, "e(x, z)", "e(x, y, 0m)", "ssh(x, y)", graph_directory * "final_e_ssh.png")

du_wind_stress = data2["du_wind_stress"]
dv_wind_stress = data2["dv_wind_stress"]
dT             = data2["dT"]
dS             = data2["dS"]
dT_flux        = data2["dT_flux"]
dkappaT_final  = data2["dkappaT_final"]
dkappaS_final  = data2["dkappaS_final"]
close(data2)

temp_tight = false

plot_variables(dT[:, j′, :], dT[:,:,grid.Nz], dT[:,:,1], xc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (x, z)", "Initial Temperature Gradient (x, y, 0m)", "Initial Temperature Gradient (x, y, 2180m)", graph_directory * "gradient_temp.png")
plot_variables(dT[:, j′, :], dT_flux[:,:,1], dT[:,:,14], xc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (x, z)", "Initial Temperature Surface Flux Gradient (x, y)", "Initial Temperature Gradient (x, y, 116m)", graph_directory * "gradient_temp_flux.png")
plot_variables(dS[:, j′, :], dS[:,:,grid.Nz], dS[:,:,1], xc, zc, xc, yc, xc, yc, "Initial Salinity Gradient (x, z)", "Initial Salinity Gradient (x, y, 0m)", "Initial Salinity Gradient (x, y, 2180m)", graph_directory * "gradient_salinity.png")

#plot_variables(dkappaT_init[:, j′, :], dkappaT_init[:,:,grid.Nz], dkappaT_init[:,:,1], xc, zc, xc, yc, xc, yc, "Initialized T Vertical Diffusivity Gradient (x, z)", "Initialized T Vertical Diffusivity Gradient  (x, y, 0m)", "Initialized T Vertical Diffusivity Gradient  (x, y, 2180m)", graph_directory * "gradient_kappaTinit.png")
#plot_variables(dkappaT_final[:, j′, :], dkappaT_final[:,:,grid.Nz], dkappaT_final[:,:,1], xc, zc, xc, yc, xc, yc, "T Vertical Diffusivity Gradient (x, z)", "T Vertical Diffusivity Gradient  (x, y, 0m)", "T Vertical Diffusivity Gradient  (x, y, 2180m)", graph_directory * "gradient_kappaT.png")

#plot_variables(dkappaS_init[:, j′, :], dkappaS_init[:,:,grid.Nz], dkappaS_init[:,:,1], xc, zc, xc, yc, xc, yc, "Initialized S Vertical Diffusivity Gradient (x, z)", "Initialized S Vertical Diffusivity Gradient  (x, y, 0m)", "Initialized S Vertical Diffusivity Gradient  (x, y, 2180m)", graph_directory * "gradient_kappaSinit.png")
#plot_variables(dkappaS_final[:, j′, :], dkappaS_final[:,:,grid.Nz], dkappaS_final[:,:,1], xc, zc, xc, yc, xc, yc, "S Vertical Diffusivity Gradient (x, z)", "S Vertical Diffusivity Gradient  (x, y, 0m)", "S Vertical Diffusivity Gradient  (x, y, 2180m)", graph_directory * "gradient_kappaS.png")

plot_variables(T_final[:, j′, :], T_final[:, :, 31], T_final[:, :, 28], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 15m)", "T(x, y, 54m)", graph_directory * "final_T_15and54m.png"; p1_min=temp_tight, p2_min=temp_tight, p3_min=temp_tight, shared=true, color=:thermal)
plot_variables(S_final[:, j′, :], S_final[:, :, 31], S_final[:, :, 28], xc, zc, xc, yc, xc, yc, "S(x, z)", "S(x, y, 15m)", "S(x, y, 54m)", graph_directory * "final_S_15and54m.png"; color=:haline)
plot_variables(dT[:, j′, :], dT[:, :, 31], dT[:, :, 28], xc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (x, z)", "Initial Temperature Gradient (x, y, 15m)", "Initial Temperature Gradient (x, y, 54m)", graph_directory * "dT_15and54m.png")
plot_variables(dS[:, j′, :], dS[:, :, 31], dS[:, :, 28], xc, zc, xc, yc, xc, yc, "Initial Salinity Gradient (x, z)", "Initial Salinity Gradient (x, y, 15m)", "Initial Salinity Gradient (x, y, 54m)", graph_directory * "dS_15and54m.png")
plot_variables(dkappaT_final[:, j′, :], dkappaT_final[:, :, 31], dkappaT_final[:, :, 28], xc, zc, xc, yc, xc, yc, "T Vertical Diffusivity Gradient (x, z)", "T Vertical Diffusivity Gradient (x, y, 15m)", "T Vertical Diffusivity Gradient (x, y, 54m)", graph_directory * "dkappa_T_15and54m.png")
plot_variables(dkappaS_final[:, j′, :], dkappaS_final[:, :, 31], dkappaS_final[:, :, 28], xc, zc, xc, yc, xc, yc, "S Vertical Diffusivity Gradient (x, z)", "S Vertical Diffusivity Gradient (x, y, 15m)", "S Vertical Diffusivity Gradient (x, y, 54m)", graph_directory * "dkappa_S_15and54m.png")


plot_variables(T_final[:, j′, :], T_final[:, :, 25], T_final[:, :, 14], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 106m)", "T(x, y, 504m)", graph_directory * "final_T_106and504m.png"; p1_min=temp_tight, p2_min=temp_tight, p3_min=temp_tight, shared=true, color=:thermal)
plot_variables(S_final[:, j′, :], S_final[:, :, 25], S_final[:, :, 14], xc, zc, xc, yc, xc, yc, "S(x, z)", "S(x, y, 106m)", "S(x, y, 504m)", graph_directory * "final_S_106and504m.png"; color=:haline)
plot_variables(dT[:, j′, :], dT[:, :, 25], dT[:, :, 14], xc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (x, z)", "Initial Temperature Gradient (x, y, 106m)", "Initial Temperature Gradient (x, y, 504m)", graph_directory * "dT_106and504m.png")
plot_variables(dS[:, j′, :], dS[:, :, 25], dS[:, :, 14], xc, zc, xc, yc, xc, yc, "Initial Salinity Gradient (x, z)", "Initial Salinity Gradient (x, y, 106m)", "Initial Salinity Gradient (x, y, 504m)", graph_directory * "dS_106and504m.png")
plot_variables(dkappaT_final[:, j′, :], dkappaT_final[:, :, 25], dkappaT_final[:, :, 14], xc, zc, xc, yc, xc, yc, "T Vertical Diffusivity Gradient (x, z)", "T Vertical Diffusivity Gradient (x, y, 106m)", "T Vertical Diffusivity Gradient (x, y, 504m)", graph_directory * "dkappa_T_106and504m.png")
plot_variables(dkappaS_final[:, j′, :], dkappaS_final[:, :, 25], dkappaS_final[:, :, 14], xc, zc, xc, yc, xc, yc, "S Vertical Diffusivity Gradient (x, z)", "S Vertical Diffusivity Gradient (x, y, 106m)", "S Vertical Diffusivity Gradient (x, y, 504m)", graph_directory * "dkappa_S_106and504m.png")


plot_variables(T_final[:, j′, :], T_final[:, :, 10], T_final[:, :, 6], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 795m)", "T(x, y, 1228m)", graph_directory * "final_T_795and1228m.png"; p1_min=temp_tight, p2_min=temp_tight, p3_min=temp_tight, shared=true, color=:thermal)
plot_variables(S_final[:, j′, :], S_final[:, :, 10], S_final[:, :, 6], xc, zc, xc, yc, xc, yc, "S(x, z)", "S(x, y, 795m)", "S(x, y, 1228m)", graph_directory * "final_S_795and1228m.png"; color=:haline)
plot_variables(dT[:, j′, :], dT[:, :, 10], dT[:, :, 6], xc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (x, z)", "Initial Temperature Gradient (x, y, 795m)", "Initial Temperature Gradient (x, y, 1228m)", graph_directory * "dT_795and1228m.png")
plot_variables(dS[:, j′, :], dS[:, :, 10], dS[:, :, 6], xc, zc, xc, yc, xc, yc, "Initial Salinity Gradient (x, z)", "Initial Salinity Gradient (x, y, 795m)", "Initial Salinity Gradient (x, y, 1228m)", graph_directory * "dS_795and1228m.png")
plot_variables(dkappaT_final[:, j′, :], dkappaT_final[:, :, 10], dkappaT_final[:, :, 6], xc, zc, xc, yc, xc, yc, "T Vertical Diffusivity Gradient (x, z)", "T Vertical Diffusivity Gradient (x, y, 795m)", "T Vertical Diffusivity Gradient (x, y, 1228m)", graph_directory * "dkappa_T_795and1228m.png")
plot_variables(dkappaS_final[:, j′, :], dkappaS_final[:, :, 10], dkappaS_final[:, :, 6], xc, zc, xc, yc, xc, yc, "S Vertical Diffusivity Gradient (x, z)", "S Vertical Diffusivity Gradient (x, y, 795m)", "S Vertical Diffusivity Gradient (x, y, 1228m)", graph_directory * "dkappa_S_795and1228m.png")


plot_variables(T_final[:, j′, :], T_final[:, :, 4], T_final[:, :, 1], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 1518m)", "T(x, y, 2076m)", graph_directory * "final_T_1518and2076m.png"; p1_min=temp_tight, p2_min=temp_tight, p3_min=temp_tight, shared=true, color=:thermal)
plot_variables(S_final[:, j′, :], S_final[:, :, 4], S_final[:, :, 1], xc, zc, xc, yc, xc, yc, "S(x, z)", "S(x, y, 1518m)", "S(x, y, 2076m)", graph_directory * "final_S_1518and2076m.png"; color=:haline)
plot_variables(dT[:, j′, :], dT[:, :, 4], dT[:, :, 1], xc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (x, z)", "Initial Temperature Gradient (x, y, 1518m)", "Initial Temperature Gradient (x, y, 2076m)", graph_directory * "dT_1518and2076m.png")
plot_variables(dS[:, j′, :], dS[:, :, 4], dS[:, :, 1], xc, zc, xc, yc, xc, yc, "Initial Salinity Gradient (x, z)", "Initial Salinity Gradient (x, y, 1518m)", "Initial Salinity Gradient (x, y, 2076m)", graph_directory * "dS_1518and2076m.png")
plot_variables(dkappaT_final[:, j′, :], dkappaT_final[:, :, 4], dkappaT_final[:, :, 1], xc, zc, xc, yc, xc, yc, "T Vertical Diffusivity Gradient (x, z)", "T Vertical Diffusivity Gradient (x, y, 1518m)", "T Vertical Diffusivity Gradient (x, y, 2076m)", graph_directory * "dkappa_T_1518and2076m.png")
plot_variables(dkappaS_final[:, j′, :], dkappaS_final[:, :, 4], dkappaS_final[:, :, 1], xc, zc, xc, yc, xc, yc, "S Vertical Diffusivity Gradient (x, z)", "S Vertical Diffusivity Gradient (x, y, 1518m)", "S Vertical Diffusivity Gradient (x, y, 2076m)", graph_directory * "dkappa_S_1518and2076m.png")


# NEED TO ADJUST TO YZ FOR THIS ONE
plot_variables_four_panels(du_wind_stress[:,:,1], dT[:,:,28], dv_wind_stress[:,:,1], dT[:,:,14], xu, yu, xc, yc, xv, yv, xc, yc, "Initial Zonal Wind Stress Gradient (x, y)", "Initial Temperature Gradient (x, y, 54m)", "Initial Meridional Wind Stress Gradient (x, y)", "Initial Temperature Gradient (x, y, 504m)", graph_directory * "gradients.png")

#plot_variables_six_panels(du_wind_stress[:,:,1], dT[:,:,28], dkappaT_final[:,:,28], dv_wind_stress[:,:,1], dT[:,:,14], dkappaT_final[:,:,14], xu, yu, xc, yc, xc, yc, xv, yv, xc, yc, xc, yc, "Initial Zonal Wind Stress Gradient (x, y)", "Initial Temperature Gradient (x, y, 54m)", "T Vertical Diffusivity Gradient (x, y, 54m)", "Initial Meridional Wind Stress Gradient (x, y)", "Initial Temperature Gradient (x, y, 504m)", "T Vertical Diffusivity Gradient (x, y, 504m)", graph_directory * "gradients.png")

z_thicknesses = zw[2:end] - zw[1:end-1]

@show z_thicknesses


plot_variables_two_panels_updown(dT[44, :, :] ./ z_thicknesses', dT[:,j′,:] ./ z_thicknesses', yc, zc, xc, zc, "(a) Initial Temperature Gradient (x=550km, y, z)", "(b) Initial Temperature Gradient (x, y=1000km, z)", "Sv °C⁻¹", "Sv °C⁻¹", graph_directory * "gradients_oceananigans_depth.png")



plot_variables_six_panels(du_wind_stress[:,:,1], dT[:,:,31], dkappaT_final[:,:,31], dv_wind_stress[:,:,1], dT[:,:,14], dkappaT_final[:,:,14], xu, yu, xc, yc, xc, yc, xv, yv, xc, yc, xc, yc,
                          "(a) Zonal Wind Stress Sensitivity (x, y)", "(c) Initial Temperature Sensitivity (x, y, 15m)", "(e) T Vertical Diffusivity Sensitivity (x, y, 15m)", "(b) Meridional Wind Stress Sensitivity (x, y)", "(d) Initial Temperature Sensitivity (x, y, 504m)", "(f) T Vertical Diffusivity Sensitivity (x, y, 504m)",
                          "Sv / m²s⁻²", "Sv / °C", "Sv / m²s⁻¹", "Sv / m²s⁻²", "Sv / °C", "Sv / m²s⁻¹",
                          graph_directory * "gradients_oceananigans_xy.png")


plot_variables_six_panels(u[:,:,31], u[:,:,14], u[:,:,4], v[:,:,31], v[:,:,14], v[:,:,4], xu, yu, xu, yu, xu, yu, xv, yv, xv, yv, xv, yv,
                          "(a) u(x, y, z=15m)", "(b) u(x, y, z=504m)", "(c) u(x, y, z=1518m)", "(d) v(x, y, z=15m)", "(e) v(x, y, z=504m)", "(f) v(x, y, z=1518m)",
                          "m s⁻¹", "m s⁻¹", "m s⁻¹", "m s⁻¹", "m s⁻¹", "m s⁻¹",
                          graph_directory * "velocities_oceananigans_xy.png"; shared = true)

plot_variables_six_panels(T_final[:,:,31], T_final[:,:,14], T_final[:,:,4], S_final[:,:,31], S_final[:,:,14], S_final[:,:,4], xc, yc, xc, yc, xc, yc, xc, yc, xc, yc, xc, yc,
                          "(a) T(x, y, z=15m)", "(b) T(x, y, z=504m)", "(c) T(x, y, z=1518m)", "(d) S(x, y, z=15m)", "(e) S(x, y, z=504m)", "(f) S(x, y, z=1518m)",
                          "°C", "°C", "°C", "m s⁻¹", "m s⁻¹", "m s⁻¹",
                          graph_directory * "tracers_oceananigans_xy.png"; color=:thermal,
                          p1_min=true, p2_min=true, p3_min=true)



plot_variables(w[45, :, :], u[:, :, grid.Nz], v[:, :, grid.Nz], yw, zw, xu, yu, xv, yv, "w(562km, y, z)", "u(x, y, 0m)", "v(x, y, 0m)", graph_directory * "final_velocities_surface.png")
plot_variables(T_final[45, :, :], T_final[:, :, 25], T_final[:, :, 14], yc, zc, xc, yc, xc, yc, "T(560km, y, z)", "T(x, y, 54m)", "T(x, y, 504m)", graph_directory * "final_T_54and504m.png"; p1_min=true, p2_min=true, p3_min=true, shared=true, color=:thermal)
#plot_variables_four_panels(dT[45, :, :], dT[:, :, 28], dT[:, :, 14], yc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (560km, y, z)", "Initial Temperature Gradient (x, y, 54m)", "Initial Temperature Gradient (x, y, 504m)", graph_directory * "dT_54and504m.png")