"""
Horizontal slices from 1/16° final snapshot (iter 9000, sim 3h).
θ, qv, u at surface / mid-level / upper.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

Rx, Ry = 4, 2
output_dir = "simulations/output/nccl_8gpu_16th_deg"
iter_str = "009000"

def load_field(name):
    tiles = []
    for r in range(Rx * Ry):
        path = os.path.join(output_dir, f"fields_rank{r}_iter{iter_str}.jld2")
        with h5py.File(path, "r") as f:
            tiles.append(np.array(f[name]).T)
    nx, ny, nz = tiles[0].shape
    g = np.zeros((Rx * nx, Ry * ny, nz), dtype=tiles[0].dtype)
    for r in range(Rx * Ry):
        g[(r//Ry)*nx:(r//Ry+1)*nx, (r%Ry)*ny:(r%Ry+1)*ny, :] = tiles[r]
    return g

print("Loading ρ, ρθ, ρqᵛ, ρu...")
rho = load_field("ρ")
rho_theta = load_field("ρθ")
rho_qv = load_field("ρqᵛ")
rho_u = load_field("ρu")

Nx, Ny, Nz = rho.shape
print(f"Global shape: {Nx}×{Ny}×{Nz}")

safe_rho = np.where(rho > 0.001, rho, 0.001)
theta = rho_theta / safe_rho
qv = rho_qv / safe_rho * 1000
u = rho_u / safe_rho

lon = np.linspace(0, 360, Nx, endpoint=False)
lat = np.linspace(-80, 80, Ny, endpoint=False)

k_sfc = 0
k_mid = Nz // 2
k_top = Nz - 5

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
fig.suptitle(f"1/16° global atmosphere — iter {iter_str} (sim 3h)\n"
             f"5760×2560×64, Δt=1.2s, 8×H100 NCCLDistributed", fontsize=14, y=0.98)

configs = [
    (theta, r'$\theta$ [K]', 'RdYlBu_r', None),
    (qv,    r'$q_v$ [g/kg]', 'YlGnBu', None),
    (u,     r'$u$ [m/s]', 'RdBu_r', 'sym'),
]

levels = [
    ('Surface (z≈0 km)', k_sfc),
    ('Mid-level (z≈15 km)', k_mid),
    ('Upper (z≈27 km)', k_top),
]

for col, (field, label, cmap, mode) in enumerate(configs):
    for row, (level_name, k) in enumerate(levels):
        ax = axes[row, col]
        data = field[:, :, k].T
        if mode == 'sym':
            vmax = np.percentile(np.abs(data), 99)
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vmax, vmax=vmax,
                               shading='nearest', rasterized=True)
        else:
            vmin, vmax = np.percentile(data, [1, 99])
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='nearest', rasterized=True)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        if row == 0:
            ax.set_title(label, fontsize=13)
        if col == 0:
            ax.set_ylabel(f'{level_name}\nlat [°]', fontsize=10)
        if row == 2:
            ax.set_xlabel('lon [°]')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('horizontal_slices_16th_iter009000.png', dpi=150, bbox_inches='tight')
print("Saved horizontal_slices_16th_iter009000.png")
