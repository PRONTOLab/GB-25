using JLD2
using CairoMakie

# Load checkpoint
checkpoint_path = joinpath(@__DIR__, "checkpoints", "ocean_tenth_degree_15day_2026-04-15T04-00-16.jld2")
@info "Loading" checkpoint_path
data = JLD2.jldopen(checkpoint_path, "r") do f
    (Nx=f["Nx"], Ny=f["Ny"], Nz=f["Nz"],
     u=f["u"], v=f["v"], T=f["T"], S=f["S"],
     time=f["time"], iteration=f["iteration"])
end

Nx, Ny, Nz = data.Nx, data.Ny, data.Nz
sim_days = round(data.time / 86400, digits=1)

# Surface slices (top level = last k index)
T_surf = data.T[:, :, Nz]
S_surf = data.S[:, :, Nz]
u_surf = data.u[:, :, Nz]
v_surf = data.v[:, :, Nz]

@info "Surface data" size(T_surf) extrema(T_surf) extrema(u_surf)

# Build lon/lat coordinate arrays
# Grid: longitude 0-360, latitude -80 to 80
λc = range(360/Nx/2, 360 - 360/Nx/2, length=Nx)  # cell centers
φc = range(-80 + 160/Ny/2, 80 - 160/Ny/2, length=Ny)

# For surface!() we need node positions (Nx+1 × Ny+1)
λn = range(0, 360, length=Nx+1)
φn = range(-80, 80, length=Ny+1)

# Spherical to Cartesian
function sphere_xyz(λ, φ; r=1.0)
    λr = deg2rad(λ)
    φr = deg2rad(φ)
    x = r * cos(φr) * cos(λr)
    y = r * cos(φr) * sin(λr)
    z = r * sin(φr)
    return x, y, z
end

# Build coordinate matrices for surface!
X = zeros(Nx+1, Ny+1)
Y = zeros(Nx+1, Ny+1)
Z = zeros(Nx+1, Ny+1)
for j in 1:Ny+1, i in 1:Nx+1
    X[i,j], Y[i,j], Z[i,j] = sphere_xyz(λn[i], φn[j])
end

@info "Plotting..."

fig = Figure(size=(1600, 1400), backgroundcolor=:black)

Label(fig[0, 1:2],
    "Ocean 1/10° — t = $(sim_days) days, iter = $(data.iteration)",
    fontsize=22, color=:white, tellwidth=false)

kw = (elevation=deg2rad(25), azimuth=deg2rad(210), aspect=:equal,
      backgroundcolor=:black)

ax_T = Axis3(fig[1, 1]; title="Temperature [°C]", titlecolor=:white, kw...)
ax_S = Axis3(fig[1, 2]; title="Salinity [psu]", titlecolor=:white, kw...)
ax_u = Axis3(fig[2, 1]; title="Zonal Velocity u [m/s]", titlecolor=:white, kw...)
ax_v = Axis3(fig[2, 2]; title="Meridional Velocity v [m/s]", titlecolor=:white, kw...)

plt_T = surface!(ax_T, X, Y, Z; color=T_surf, colormap=:thermal, colorrange=(0, 30), shading=NoShading)
plt_S = surface!(ax_S, X, Y, Z; color=S_surf, colormap=:haline, colorrange=(28, 31), shading=NoShading)
plt_u = surface!(ax_u, X, Y, Z; color=u_surf, colormap=:balance, colorrange=(-10, 10), shading=NoShading)
plt_v = surface!(ax_v, X, Y, Z; color=v_surf, colormap=:balance, colorrange=(-8, 8), shading=NoShading)

for ax in [ax_T, ax_S, ax_u, ax_v]
    hidedecorations!(ax)
    hidespines!(ax)
end

Colorbar(fig[3, 1], plt_T; label="T [°C]", labelcolor=:white, tickcolor=:white, ticklabelcolor=:white, vertical=false, flipaxis=false)
Colorbar(fig[3, 2], plt_S; label="S [psu]", labelcolor=:white, tickcolor=:white, ticklabelcolor=:white, vertical=false, flipaxis=false)

outpath = joinpath(@__DIR__, "checkpoints", "tenth_degree_sphere.png")
save(outpath, fig, px_per_unit=2)
@info "Saved" outpath filesize(outpath)
