"""
Side-by-side visualization of 1/16° and 1/24° final snapshots.
Surface θ, qv, u for both resolutions.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

def load_field(output_dir, name, iter_str, Rx, Ry):
    tiles = []
    for r in range(Rx * Ry):
        path = os.path.join(output_dir, f"fields_rank{r}_iter{iter_str}.jld2")
        with h5py.File(path, "r") as f:
            tiles.append(np.array(f[name]).T)
    nx, ny, nz = tiles[0].shape
    global_field = np.zeros((Rx * nx, Ry * ny, nz), dtype=tiles[0].dtype)
    for r in range(Rx * Ry):
        ix = r // Ry
        iy = r % Ry
        global_field[ix*nx:(ix+1)*nx, iy*ny:(iy+1)*ny, :] = tiles[r]
    return global_field

Rx, Ry = 4, 2

# Load 1/24° final snapshot
print("Loading 1/24° (iter 22500)...")
dir_24 = "simulations/output/nccl_8gpu_24th_deg"
rho_24 = load_field(dir_24, "ρ", "022500", Rx, Ry)
rho_theta_24 = load_field(dir_24, "ρθ", "022500", Rx, Ry)
rho_qv_24 = load_field(dir_24, "ρqᵛ", "022500", Rx, Ry)
rho_u_24 = load_field(dir_24, "ρu", "022500", Rx, Ry)

# Load 1/16° final snapshot
print("Loading 1/16° (iter 9000)...")
dir_16 = "simulations/output/nccl_8gpu_16th_deg"
rho_16 = load_field(dir_16, "ρ", "009000", Rx, Ry)
rho_theta_16 = load_field(dir_16, "ρθ", "009000", Rx, Ry)
rho_qv_16 = load_field(dir_16, "ρqᵛ", "009000", Rx, Ry)
rho_u_16 = load_field(dir_16, "ρu", "009000", Rx, Ry)

def derive(rho, rho_theta, rho_qv, rho_u):
    safe = np.where(rho > 0.001, rho, 0.001)
    return rho_theta / safe, rho_qv / safe * 1000, rho_u / safe

theta_24, qv_24, u_24 = derive(rho_24, rho_theta_24, rho_qv_24, rho_u_24)
theta_16, qv_16, u_16 = derive(rho_16, rho_theta_16, rho_qv_16, rho_u_16)

lon_24 = np.linspace(0, 360, theta_24.shape[0], endpoint=False)
lat_24 = np.linspace(-80, 80, theta_24.shape[1], endpoint=False)
lon_16 = np.linspace(0, 360, theta_16.shape[0], endpoint=False)
lat_16 = np.linspace(-80, 80, theta_16.shape[1], endpoint=False)

# Plot: 3 fields × 2 resolutions × 2 levels = 12 panels
k_sfc = 0
k_mid = 32

fig, axes = plt.subplots(4, 3, figsize=(22, 16))
fig.suptitle("1/16° vs 1/24° global atmosphere\n"
             "1/16°: 3h sim, Δt=1.2s | 1/24°: 5h sim, Δt=0.8s | 8×H100 NCCLDistributed",
             fontsize=14, y=0.98)

configs = [
    (r'$\theta$ [K]', 'RdYlBu_r', None),
    (r'$q_v$ [g/kg]', 'YlGnBu', None),
    (r'$u$ [m/s]', 'RdBu_r', 'sym'),
]

rows = [
    ("1/16° surface", lon_16, lat_16, [theta_16[:,:,k_sfc], qv_16[:,:,k_sfc], u_16[:,:,k_sfc]]),
    ("1/16° mid-level", lon_16, lat_16, [theta_16[:,:,k_mid], qv_16[:,:,k_mid], u_16[:,:,k_mid]]),
    ("1/24° surface", lon_24, lat_24, [theta_24[:,:,k_sfc], qv_24[:,:,k_sfc], u_24[:,:,k_sfc]]),
    ("1/24° mid-level", lon_24, lat_24, [theta_24[:,:,k_mid], qv_24[:,:,k_mid], u_24[:,:,k_mid]]),
]

for row_idx, (row_label, lon, lat, fields) in enumerate(rows):
    for col_idx, (field, (label, cmap, mode)) in enumerate(zip(fields, configs)):
        ax = axes[row_idx, col_idx]
        data = field.T

        if mode == 'sym':
            vmax = np.percentile(np.abs(data), 99)
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vmax, vmax=vmax,
                               shading='nearest', rasterized=True)
        else:
            vmin, vmax = np.percentile(data, [1, 99])
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='nearest', rasterized=True)

        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        if row_idx == 0:
            ax.set_title(label, fontsize=13)
        if col_idx == 0:
            ax.set_ylabel(f'{row_label}\nlat [°]', fontsize=10)
        if row_idx == 3:
            ax.set_xlabel('lon [°]')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('comparison_16th_vs_24th.png', dpi=150, bbox_inches='tight')
print("Saved comparison_16th_vs_24th.png")
