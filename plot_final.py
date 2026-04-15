"""
Quick visualization of final snapshots: just θ at surface and mid-level
for both 1/16° and 1/24°. Loads only ρ and ρθ (2 fields per resolution).
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

Rx, Ry = 4, 2

def load_field(output_dir, name, iter_str):
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

print("Loading 1/16° ρ, ρθ...")
rho_16 = load_field("simulations/output/nccl_8gpu_16th_deg", "ρ", "009000")
rt_16 = load_field("simulations/output/nccl_8gpu_16th_deg", "ρθ", "009000")

print("Loading 1/24° ρ, ρθ...")
rho_24 = load_field("simulations/output/nccl_8gpu_24th_deg", "ρ", "022500")
rt_24 = load_field("simulations/output/nccl_8gpu_24th_deg", "ρθ", "022500")

theta_16 = rt_16 / np.where(rho_16 > 0.001, rho_16, 0.001)
theta_24 = rt_24 / np.where(rho_24 > 0.001, rho_24, 0.001)

lon_16 = np.linspace(0, 360, theta_16.shape[0], endpoint=False)
lat_16 = np.linspace(-80, 80, theta_16.shape[1], endpoint=False)
lon_24 = np.linspace(0, 360, theta_24.shape[0], endpoint=False)
lat_24 = np.linspace(-80, 80, theta_24.shape[1], endpoint=False)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("Final snapshots: potential temperature θ [K]\n"
             "1/16° (3h sim, Δt=1.2) vs 1/24° (5h sim, Δt=0.8)", fontsize=14)

panels = [
    (axes[0,0], "1/16° surface", lon_16, lat_16, theta_16[:,:,0]),
    (axes[0,1], "1/16° mid-level (k=32)", lon_16, lat_16, theta_16[:,:,32]),
    (axes[1,0], "1/24° surface", lon_24, lat_24, theta_24[:,:,0]),
    (axes[1,1], "1/24° mid-level (k=32)", lon_24, lat_24, theta_24[:,:,32]),
]

for ax, title, lon, lat, data in panels:
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax.pcolormesh(lon, lat, data[:,:].T, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                       shading='nearest', rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
    ax.set_ylabel('lat [°]')
    ax.set_xlabel('lon [°]')

plt.tight_layout()
plt.savefig('final_snapshots_theta.png', dpi=150, bbox_inches='tight')
print("Saved final_snapshots_theta.png")
