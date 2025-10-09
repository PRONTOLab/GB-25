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

graph_directory = "run_abernathy_model_ad_900steps_mildVisc_CenteredOrder4_partialCell/"

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

halo_size = 4

const Lx = 1000kilometers # zonal domain length [m]
const Ly = 2000kilometers # meridional domain length [m]
const Lz = sum(Δz_center)

# full ridge function:
function ridge_function(x, y)
    zonal = (Lz+300)exp(-(x - Lx/2)^2/(1e6kilometers))
    gap   = 1 - 0.5(tanh((y - (Ly/6))/1e5) - tanh((y - (Ly/2))/1e5))
    return zonal * gap - Lz
end

function make_grid(architecture, Nx, Ny, Nz, Δz_center)
    z_faces = vcat([-Lz], -Lz .+ cumsum(Δz_center))
    z_faces[Nz+1] = 0

    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    # Make into a ridge array:
    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, ridge_function)

    grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(ridge))
    return grid
end

grid = make_grid(CPU(), Nx, Ny, Nz, Δz_center)

bottom_height = data1["bottom_height"]
T_init        = data1["T_init"]
e_init        = data1["e_init"]
wind_stress   = data1["u_wind_stress"]

data2 = jldopen(graph_directory * "data_final.jld2", "r")

T_final = data2["T_final"]
e_final = data2["e_final"]
ssh     = data2["ssh"]

u = data2["u"]
v = data2["v"]
w = data2["w"]

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
                        filename)

    wmax = max(1e-9, maximum(abs, p1_xz))
    umax = max(1e-9, maximum(abs, p2_xy))
    vmax = max(1e-9, maximum(abs, p3_xy))

    wlims = (-wmax, wmax) .* 0.8
    ulims = (-umax, umax) .* 0.8
    vlims = (-vmax, vmax) .* 0.8

    wlevels = vcat([-wmax], range(wlims[1], wlims[2], length = 31), [wmax])
    ulevels = vcat([-umax], range(ulims[1], ulims[2], length = 31), [umax])
    vlevels = vcat([-vmax], range(vlims[1], vlims[2], length = 31), [vmax])

    xlims = (0, grid.Lx) .* 1e-3
    ylims = (0, grid.Ly) .* 1e-3
    zlims = (-grid.Lz, 0)

    p1_xz_plot = contourf(x1 * 1e-3, z1, p1_xz',
        xlabel = "x (km)",
        ylabel = "z (m)",
        aspectratio = 0.05,
        linewidth = 0,
        levels = wlevels,
        clims = wlims,
        xlims = xlims,
        ylims = zlims,
        color = :balance)

    p2_xy_plot = contourf(x2 * 1e-3, y2 * 1e-3, p2_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = ulevels,
        clims = ulims,
        xlims = xlims,
        ylims = ylims,
        color = :balance)

    p3_xy_plot = contourf(x3 * 1e-3, y3 * 1e-3, p3_xy',
        xlabel = "x (km)",
        ylabel = "y (km)",
        aspectratio = :equal,
        linewidth = 0,
        levels = vlevels,
        clims = vlims,
        xlims = xlims,
        ylims = ylims,
        color = :balance)

    layout = @layout [upper_slice_plot{0.2h}
        Plots.grid(1, 2)]

    plot(p1_xz_plot, p2_xy_plot, p3_xy_plot, layout = layout, size = (1200, 1200), title = [p1_title p2_title p3_title])

    savefig(filename)
end

plot_variables(w[:, j′, :], u[:, :, grid.Nz], v[:, :, grid.Nz], xw, zw, xu, yu, xv, yv, "w(x, z)", "u(x, y, 0m)", "v(x, y, 0m)", graph_directory * "final_velocities_surface.png")
plot_variables(w[:, j′, :], u[:, :, 1], v[:, :, 1], xw, zw, xu, yu, xv, yv, "w(x, z)", "u(x, y, 2180m)", "v(x, y, 2180m)", graph_directory * "final_velocities_bottom.png")
plot_variables(u[:, j′, :], u[:, :, 14], v[:, :, 14], xu, zu, xu, yu, xv, yv, "u(x, z)", "u(x, y, 533m)", "v(x, y, 533m)", graph_directory * "final_velocities_533m.png")
plot_variables(v[:, j′, :], u[:, :, 25], v[:, :, 25], xv, zv, xu, yu, xv, yv, "v(x, z)", "u(x, y, 116m)", "v(x, y, 116m)", graph_directory * "final_velocities_116m.png")


plot_variables(T_final[:, j′, :], T_final[:, :, grid.Nz], T_final[:, :, 25], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 0m)", "T(x, y, 116m)", graph_directory * "final_T_0to116m.png")
plot_variables(T_final[:, j′, :], T_final[:, :, 16], T_final[:, :, 1], xc, zc, xc, yc, xc, yc, "T(x, z)", "T(x, y, 533m)", "T(x, y, 2180m)", graph_directory * "final_T_533to2180m.png")

plot_variables(e_final[:, j′, :], e_final[:,:,grid.Nz], ssh[:,:,1], xc, zc, xc, yc, xc, yc, "e(x, z)", "e(x, y, 0m)", "ssh(x, y)", graph_directory * "final_e_ssh.png")

du_wind_stress = data2["du_wind_stress"]
dv_wind_stress = data2["dv_wind_stress"]
dT             = data2["dT"]
close(data2)

plot_variables(dT[:, j′, :], du_wind_stress[:,:,1], dv_wind_stress[:,:,1], xc, zc, xu, yu, xv, yv, "Initial Temperature Gradient (x, z)", "Initial Zonal Wind Stress Gradient (x, y)", "Initial Meridional Wind Stress Gradient (x, y)", graph_directory * "gradient_wind_stress.png")
plot_variables(dT[:, j′, :], dT[:,:,grid.Nz], dT[:,:,1], xc, zc, xc, yc, xc, yc, "Initial Temperature Gradient (x, z)", "Initial Temperature Gradient (x, y, 0m)", "Initial Temperature Gradient (x, y, 2180m)", graph_directory * "gradient_temp.png")