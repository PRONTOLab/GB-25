using Makie
using CairoMakie

colors = Makie.wong_colors()
ocean_color = colors[1]
atmos_color = colors[2]

ocean_alps = [
    32   56.475454 6128*3056   ;
    288  50.847629 18416*9200  ;
    512  51.785820 24560*12272 ;
    968  53.128941 33776*16880 ;
    2048 53.963027 49136*24560 ;
    3872 52.206009 67568*33776 ;
    8192 56.977438 98288*49136 ;
]

x_ocean_alps  = ocean_alps[:, 1]
t_ocean_alps  = ocean_alps[:, 2] ./ 256
gp_ocean_alps = ocean_alps[:, 3]
y_ocean_alps = gp_ocean_alps ./ t_ocean_alps ./ x_ocean_alps

atmos_alps = [
    128    152.154855    12280*6136 ;
    288    181.078936    18424*9208 ;
    512    149.214914    24568*12280;
    2048   151.351572    49144*24568;
    3872   146.962295    67576*33784;
]

x_atmos_alps  = atmos_alps[:, 1]
t_atmos_alps  = atmos_alps[:, 2] ./ 256
gp_atmos_alps = atmos_alps[:, 3]
y_atmos_alps = gp_atmos_alps ./ t_atmos_alps ./ x_atmos_alps

fig = Figure(size = (600, 400))
ax = Axis(
    fig[1, 1],
    # title = "Weak scaling",
    xlabel = "Number of GPUs",
    xscale = log2,
    xticks = unique!([x_ocean_alps; x_atmos_alps]),
    ylabel = "Grid points per time per GPU [s⁻¹]",
)


# Lines for the datapoints
lines!(ax, x_ocean_alps, y_ocean_alps; linewidth=2, color=ocean_color)
lines!(ax, x_atmos_alps, y_atmos_alps; linewidth=2, color=atmos_color)
# Scatter plot for the datapoints
scatter!(ax, x_ocean_alps, y_ocean_alps; markersize = 10, color=ocean_color, marker=:circle, label="Ocean")
scatter!(ax, x_atmos_alps, y_atmos_alps; markersize = 10, color=atmos_color, marker=:rect,   label="Atmosphere")

# Ideal scaling lines
hlines!(ax, y_ocean_alps[1], linestyle = :dash, color=ocean_color, linewidth=2)
hlines!(ax, y_atmos_alps[1], linestyle = :dash, color=atmos_color, linewidth=2)
# Fake line not plotted, just for having an entry in the legend with a "neutral" color.
hlines!(ax, -1, linestyle = :dash, color=:grey, linewidth=2, label="Ideal weak scaling")

# Set y axis limits
ylims!(ax, 0, maximum([y_ocean_alps; y_atmos_alps]) .* 1.1)

# add labels slightly above each point
text!(
    ax,
    x_ocean_alps,
    y_ocean_alps,
    text = string.(round.(y_ocean_alps ./ y_ocean_alps[1]; digits=2)),
    align = (:center, :bottom),
    offset = (0, 5)  # shift upward a bit
)
text!(
    ax,
    x_atmos_alps,
    y_atmos_alps,
    text = string.(round.(y_atmos_alps ./ y_atmos_alps[1]; digits=2)),
    align = (:center, :bottom),
    offset = (0, 5)  # shift upward a bit
)

# Legend
axislegend(ax; position=:rb)

save("weak-scaling.png", fig)
