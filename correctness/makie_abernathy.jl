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

du_wind_stress = data2["du_wind_stress"]
dv_wind_stress = data2["dv_wind_stress"]
dT             = data2["dT"]
dS             = data2["dS"]
dT_flux        = data2["dT_flux"]
dkappaT_final  = data2["dkappaT_final"]
dkappaS_final  = data2["dkappaS_final"]
close(data2)

using GLMakie

function plot_variables_six_panels_3x2(
    p1_xy, p2_xy, p3_xy, p4_xy, p5_xy, p6_xy,
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6,
    p1_title, p2_title, p3_title, p4_title, p5_title, p6_title,
    p1_cbt, p2_cbt, p3_cbt, p4_cbt, p5_cbt, p6_cbt,  # colorbar titles
    filename;
    p1_min=false, p2_min=false, p3_min=false, p4_min=false, p5_min=false, p6_min=false,
    shared=false, color=:balance)

    # Compute mins and maxes
    pmins = [minimum(p) for p in (p1_xy,p2_xy,p3_xy,p4_xy,p5_xy,p6_xy)]
    pmaxs = [maximum(p) for p in (p1_xy,p2_xy,p3_xy,p4_xy,p5_xy,p6_xy)]

    for i in 1:6
        if !( (p1_min,p2_min,p3_min,p4_min,p5_min,p6_min)[i] )
            mmax = max(pmaxs[i], -pmins[i])
            pmaxs[i] = mmax
            pmins[i] = -mmax
        end
    end

    if shared
        mmax = maximum(abs.(vcat(pmins, pmaxs)))
        pmins .= -mmax
        pmaxs .=  mmax
    end

    # Bounds for levels, reduced by 20% visually just like Plots
    plims   = [(pmins[i], pmaxs[i]) .* 0.8 for i in 1:6]
    levels  = [[range(plims[i][1], plims[i][2], length=62)...] for i in 1:6]

    # scaled coordinates (km)
    xs = [x1,x2,x3,x4,x5,x6] .|> x -> x .* 1e-3
    ys = [y1,y2,y3,y4,y5,y6] .|> y -> y .* 1e-3
    ps = [p1_xy,p2_xy,p3_xy,p4_xy,p5_xy,p6_xy]
    titles = [p1_title, p2_title, p3_title, p4_title, p5_title, p6_title]
    cb_titles = [p1_cbt, p2_cbt, p3_cbt, p4_cbt, p5_cbt, p6_cbt]

    # Build Figure (3 rows × 2 columns, but each cell has axis + colorbar)
    fig = Figure(resolution=(1500,2250))

    # Plot each panel
    idx = 1
    for r in 1:3
        for c in 1:2
            ax = Axis(fig[r, 2c-1],
                title = titles[idx],
                xlabel = (idx in (5,6) ? "x (km)" : ""),
                ylabel = (c == 1 ? "y (km)" : ""),
                aspect = 0.75,
                titlesize = 22,
                xlabelsize = 20,
                ylabelsize = 20,
                xticklabelsize = 20,
                yticklabelsize = 20,
                leftspinevisible = false,
                rightspinevisible = false,
                topspinevisible = false,
                bottomspinevisible = false)

            # contourf: Makie wants z as matrix matching x/y dims. You already transposed, so we use parent arrays as-is.
            # (Makie does not auto-transpose like Plots.jl did)
            cont = contourf!(ax, xs[idx], ys[idx], ps[idx],
                    colormap=color,
                    levels=levels[idx],
                    colorscale=plims[idx])

            # Add colorbar in right-hand cell for this panel
            Colorbar(fig[r, 2c];
                colormap = to_colormap(color),
                limits = plims[idx],
                label = cb_titles[idx],
                labelsize = 20,
                ticklabelsize = 16,
                tickformat = xs -> map(x -> @sprintf("%.2e", x), xs)
            )

            idx += 1
        end
    end

    save(filename, fig)
    return fig
end


plot_variables_six_panels_3x2(du_wind_stress[:,:,1], dv_wind_stress[:,:,1], dT[:,:,31], dT[:,:,14], dkappaT_final[:,:,31], dkappaT_final[:,:,14], xu, yu, xv, yv, xc, yc, xc, yc, xc, yc, xc, yc,
                          "(a) Zonal Wind Stress Sensitivity (x, y)", "(b) Meridional Wind Stress Sensitivity (x, y)", "(c) Initial Temperature Sensitivity (x, y, 15m)", "(d) Initial Temperature Sensitivity (x, y, 504m)", "(e) T Vertical Diffusivity Sensitivity (x, y, 15m)", "(f) T Vertical Diffusivity Sensitivity (x, y, 504m)",
                          "Sv / m²s⁻²", "Sv / m²s⁻²", "Sv / °C", "Sv / °C", "Sv / m²s⁻¹", "Sv / m²s⁻¹",
                          graph_directory * "gradients_oceananigans_xy.png")