"""
Visualize horizontal slices from the 1/24° NCCL distributed output.
Assembles rank tiles into a global field and plots at chosen z-levels.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

output_dir = "simulations/output/nccl_8gpu_24th_deg"
iter_str = "015000"

Rx, Ry = 4, 2

def load_field(name, iter_str):
    tiles = []
    for r in range(Rx * Ry):
        path = os.path.join(output_dir, f"fields_rank{r}_iter{iter_str}.jld2")
        with h5py.File(path, "r") as f:
            # h5py reads in row-major (C) order; Julia stored in column-major.
            # JLD2/HDF5 stores Julia's (Nx, Ny, Nz) as (Nz, Ny, Nx) in HDF5.
            # Transpose back to Julia order.
            arr = np.array(f[name])
            tiles.append(arr.T)  # (Nz, Ny, Nx) -> (Nx, Ny, Nz)

    nx, ny, nz = tiles[0].shape
    print(f"  tile shape (after transpose): ({nx}, {ny}, {nz})")

    # Oceananigans Distributed rank ordering: rank = ix * Ry + iy
    # (x varies slowest, y varies fastest)
    # Try both orderings and pick the one that looks right
    global_field = np.zeros((Rx * nx, Ry * ny, nz), dtype=tiles[0].dtype)
    for r in range(Rx * Ry):
        # Standard Oceananigans: rank = local_index in MPI communicator
        # Partition(Rx, Ry) typically maps rank -> (ix, iy) with
        # ix = rank ÷ Ry, iy = rank % Ry
        ix = r // Ry
        iy = r % Ry
        global_field[ix*nx:(ix+1)*nx, iy*ny:(iy+1)*ny, :] = tiles[r]

    return global_field

print("Loading ρ...")
rho = load_field("ρ", iter_str)
Nx, Ny, Nz = rho.shape
print(f"Global shape: {rho.shape}")

print("Loading ρθ...")
rho_theta = load_field("ρθ", iter_str)

print("Loading ρqᵛ...")
rho_qv = load_field("ρqᵛ", iter_str)

print("Loading ρu...")
rho_u = load_field("ρu", iter_str)

# Derived fields
safe_rho = np.where(rho > 0.001, rho, 0.001)
theta = rho_theta / safe_rho
qv = rho_qv / safe_rho * 1000  # g/kg
u = rho_u / safe_rho

lon = np.linspace(0, 360, Nx, endpoint=False)
lat = np.linspace(-80, 80, Ny, endpoint=False)

# Three z-levels
k_sfc = 0
k_mid = Nz // 2
k_top = Nz - 5

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
fig.suptitle(f"1/24° global atmosphere — iter {iter_str} (sim 3h20m)\n"
             f"8640×3840×64, Δt=0.8s, 8×H100 NCCLDistributed", fontsize=14, y=0.98)

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
        data = field[:, :, k].T  # (lon, lat) -> (lat, lon) for pcolormesh

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
outpath = 'horizontal_slices_iter015000.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
