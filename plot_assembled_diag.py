"""Plot derived fields at surface from assembled 1/16° checkpoint."""
import numpy as np
import matplotlib.pyplot as plt
import h5py

path = "simulations/initial_conditions/sixteenth_deg_12h_assembled.jld2"
f = h5py.File(path, "r")
raw = {}
for name in ["ρ", "ρu", "ρv", "ρw", "ρθ", "ρqᵛ", "micro_ρqᶜˡ", "micro_ρqᶜⁱ", "ρqʳ"]:
    raw[name] = np.array(f[name]).T  # (Nz,Ny,Nx) → (Nx,Ny,Nz)
f.close()

Nλ, Nφ_c, Nz = raw["ρ"].shape
rho = raw["ρ"]
safe_rho = np.where(rho > 0.001, rho, 0.001)

# Derived
theta = raw["ρθ"] / safe_rho
qv = raw["ρqᵛ"] / safe_rho * 1000
u = raw["ρu"] / safe_rho
v = raw["ρv"][:, :Nφ_c, :] / safe_rho
wspd = np.sqrt(u**2 + v**2)
w = raw["ρw"][:, :, :Nz] / safe_rho
T = theta * (rho * 287.0 * theta / 1e5) ** (2.0/7.0 / (1 - 2.0/7.0))  # approximate
qcl = raw["micro_ρqᶜˡ"] / safe_rho * 1000
qci = raw["micro_ρqᶜⁱ"] / safe_rho * 1000
qr = raw["ρqʳ"] / safe_rho * 1000

lon = np.linspace(0, 360, Nλ, endpoint=False)
lat = np.linspace(-80, 80, Nφ_c, endpoint=False)

k = 0  # surface

fig, axes = plt.subplots(4, 3, figsize=(22, 16))
fig.suptitle("Assembled 1/16° (12h) — derived fields at surface", fontsize=14)

configs = [
    # Row 1: dynamics
    ("θ [K]", theta[:,:,k].T, "RdYlBu_r", (250, 315)),
    ("qv [g/kg]", qv[:,:,k].T, "YlGnBu", (0, 32)),
    ("wind speed [m/s]", wspd[:,:,k].T, "hot_r", (0, 30)),
    # Row 2: momentum
    ("u [m/s]", u[:,:,k].T, "RdBu_r", None),
    ("v [m/s]", v[:,:,k].T, "RdBu_r", None),
    ("w [m/s]", w[:,:,k].T, "RdBu_r", (-0.5, 0.5)),
    # Row 3: cloud/precip
    ("qcl [g/kg]", qcl[:,:,k].T, "Blues", (0, 1)),
    ("qci [g/kg]", qci[:,:,k].T, "Purples", (0, 1)),
    ("qr [g/kg]", qr[:,:,k].T, "Oranges", (0, 1)),
    # Row 4: base state
    ("ρ [kg/m³]", rho[:,:,k].T, "viridis", None),
    ("ρθ [kg·K/m³]", raw["ρθ"][:,:,k].T, "RdYlBu_r", None),
    ("ρqv [kg/m³]", raw["ρqᵛ"][:,:,k].T, "YlGnBu", None),
]

for idx, (name, data, cmap, clim) in enumerate(configs):
    ax = axes[idx // 3, idx % 3]
    if clim is not None:
        if clim[0] == -clim[1] or (isinstance(clim, tuple) and len(clim) == 2):
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=clim[0], vmax=clim[1],
                               shading='nearest', rasterized=True)
        else:
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=clim[0], vmax=clim[1],
                               shading='nearest', rasterized=True)
    elif "u [" in name or "v [" in name:
        vmax = np.percentile(np.abs(data), 99)
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='nearest', rasterized=True)
    else:
        vmin, vmax = np.percentile(data, [1, 99])
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='nearest', rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(name)
    if idx % 3 == 0:
        ax.set_ylabel('lat [°]')
    if idx >= 9:
        ax.set_xlabel('lon [°]')

plt.tight_layout()
plt.savefig('assembled_derived_fields.png', dpi=150, bbox_inches='tight')
print("Saved assembled_derived_fields.png")
