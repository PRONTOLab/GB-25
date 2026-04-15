"""Reassemble 4-rank distributed output and plot the final state."""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("simulations/output/nccl_4gpu_3h")

# Partition: Rx=2, Ry=2
# Rank layout (Oceananigans convention): x-major ordering
#   rank 0: x=[0, Nx/2), y=[0, Ny/2)
#   rank 1: x=[Nx/2, Nx), y=[0, Ny/2)
#   rank 2: x=[0, Nx/2), y=[Ny/2, Ny)
#   rank 3: x=[Nx/2, Nx), y=[Ny/2, Ny)

def load_and_assemble(label, field_name):
    """Load a field from all 4 ranks and stitch into global array."""
    ranks = []
    for r in range(4):
        fname = output_dir / f"state_rank{r}_{label}.jld2"
        with h5py.File(fname, "r") as f:
            ranks.append(f[field_name][:])

    # h5py reads Julia column-major (Nλ, Nφ, Nz) as C-order (Nz, Nφ, Nλ)
    # Rx=2 (x-split), Ry=2 (y-split)
    # Oceananigans: rank = ix + Rx * iy
    # rank 0: ix=0, iy=0  -> x_lo, y_lo
    # rank 1: ix=1, iy=0  -> x_hi, y_lo
    # rank 2: ix=0, iy=1  -> x_lo, y_hi
    # rank 3: ix=1, iy=1  -> x_hi, y_hi
    # In h5py order: axis 0=Nz, axis 1=Nφ, axis 2=Nλ

    # Oceananigans: index2rank(i,j,k) = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)
    # With Rx=2, Ry=2, Rz=1:
    #   rank 0: i=1, j=1 → x_lo, y_lo
    #   rank 1: i=1, j=2 → x_lo, y_hi
    #   rank 2: i=2, j=1 → x_hi, y_lo
    #   rank 3: i=2, j=2 → x_hi, y_hi
    # h5py axes: (Nz, Nφ, Nλ) → y-cat on axis 1, x-cat on axis 2

    # x_hi ranks (i=2: ranks 2,3) store x-data in reverse order
    left  = np.concatenate([ranks[0], ranks[1]], axis=1)                      # y-cat, i=1
    right = np.concatenate([ranks[2][:,:,::-1], ranks[3][:,:,::-1]], axis=1)  # y-cat, i=2 (x-flipped)
    global_field = np.concatenate([left, right], axis=2)                       # x-cat → full globe
    # Result shape: (Nz, Nφ, Nλ)
    return global_field

# Load final and IC states
print("Loading final state...")
T_final = load_and_assemble("final", "T")
qv_final = load_and_assemble("final", "qᵛ")
qcl_final = load_and_assemble("final", "qᶜˡ")
qci_final = load_and_assemble("final", "qᶜⁱ")
qr_final = load_and_assemble("final", "qʳ")
u_final = load_and_assemble("final", "u")
w_final = load_and_assemble("final", "w")
rho_final = load_and_assemble("final", "ρ")

print("Loading IC state...")
T_ic = load_and_assemble("ic", "T")
qv_ic = load_and_assemble("ic", "qᵛ")

# Shape is (Nz, Nφ, Nλ) in h5py C-order
print(f"Global field shape (Nz, Nφ, Nλ): {T_final.shape}")

Nz, Nφ, Nλ = T_final.shape
lon = np.linspace(0, 360, Nλ, endpoint=False)
lat = np.linspace(-70, 70, Nφ, endpoint=False)
z = np.linspace(0, 30000, Nz, endpoint=False)

k_surf = 0
k_mid = 32
k_top = 55

# ── Figure 1: Multi-panel horizontal slices at different levels ──
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

im0 = axes[0, 0].pcolormesh(lon, lat, T_final[k_surf, :, :], cmap="RdBu_r", shading="auto")
axes[0, 0].set_title(f"Temperature (K) — z={z[k_surf]:.0f}m")
axes[0, 0].set_ylabel("Latitude")
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].pcolormesh(lon, lat, T_final[k_mid, :, :], cmap="RdBu_r", shading="auto")
axes[0, 1].set_title(f"Temperature (K) — z={z[k_mid]:.0f}m")
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].pcolormesh(lon, lat, u_final[k_mid, :, :], cmap="RdBu_r", shading="auto",
                              vmin=-60, vmax=60)
axes[0, 2].set_title(f"Zonal wind u (m/s) — z={z[k_mid]:.0f}m")
plt.colorbar(im2, ax=axes[0, 2])

im3 = axes[1, 0].pcolormesh(lon, lat, qv_final[k_surf, :, :] * 1000, cmap="YlGnBu", shading="auto")
axes[1, 0].set_title(f"Specific humidity qv (g/kg) — z={z[k_surf]:.0f}m")
axes[1, 0].set_ylabel("Latitude")
axes[1, 0].set_xlabel("Longitude")
plt.colorbar(im3, ax=axes[1, 0])

qcl_plot = np.clip(qcl_final[k_mid, :, :] * 1e6, 0, None)
im4 = axes[1, 1].pcolormesh(lon, lat, qcl_plot, cmap="Blues", shading="auto")
axes[1, 1].set_title(f"Cloud liquid (mg/kg) — z={z[k_mid]:.0f}m")
axes[1, 1].set_xlabel("Longitude")
plt.colorbar(im4, ax=axes[1, 1])

qci_plot = np.clip(qci_final[k_top, :, :] * 1e6, 0, None)
im5 = axes[1, 2].pcolormesh(lon, lat, qci_plot, cmap="Purples", shading="auto")
axes[1, 2].set_title(f"Cloud ice (mg/kg) — z={z[k_top]:.0f}m")
axes[1, 2].set_xlabel("Longitude")
plt.colorbar(im5, ax=axes[1, 2])

fig.suptitle("Quarter-degree atmosphere — final state (1440x560x64, sim_time=3.1h, 4 GPU NCCL)", fontsize=14)
plt.tight_layout()
plt.savefig("final_state_horizontal.png", dpi=150, bbox_inches="tight")
print("Saved final_state_horizontal.png")

# ── Figure 2: Latitude-height cross-sections (zonal mean) ──
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))

# Zonal mean over Nλ (axis 2) → shape (Nz, Nφ)
T_zm = np.mean(T_final, axis=2)
im = axes2[0, 0].pcolormesh(lat, z / 1000, T_zm, cmap="RdBu_r", shading="auto")
axes2[0, 0].set_title("Zonal-mean Temperature (K)")
axes2[0, 0].set_ylabel("Height (km)")
plt.colorbar(im, ax=axes2[0, 0])

u_zm = np.mean(u_final, axis=2)
im = axes2[0, 1].pcolormesh(lat, z / 1000, u_zm, cmap="RdBu_r", shading="auto", vmin=-60, vmax=60)
axes2[0, 1].set_title("Zonal-mean Zonal Wind (m/s)")
plt.colorbar(im, ax=axes2[0, 1])

qv_zm = np.mean(qv_final, axis=2) * 1000
im = axes2[1, 0].pcolormesh(lat, z / 1000, qv_zm, cmap="YlGnBu", shading="auto")
axes2[1, 0].set_title("Zonal-mean Specific Humidity (g/kg)")
axes2[1, 0].set_ylabel("Height (km)")
axes2[1, 0].set_xlabel("Latitude")
plt.colorbar(im, ax=axes2[1, 0])

qtot_zm = np.mean(np.clip(qcl_final, 0, None) + np.clip(qci_final, 0, None), axis=2) * 1e6
im = axes2[1, 1].pcolormesh(lat, z / 1000, qtot_zm, cmap="Blues", shading="auto")
axes2[1, 1].set_title("Zonal-mean Total Cloud (mg/kg)")
axes2[1, 1].set_xlabel("Latitude")
plt.colorbar(im, ax=axes2[1, 1])

fig2.suptitle("Quarter-degree atmosphere — zonal-mean cross-sections (t=3.1h)", fontsize=14)
plt.tight_layout()
plt.savefig("final_state_zonal_mean.png", dpi=150, bbox_inches="tight")
print("Saved final_state_zonal_mean.png")

# ── Figure 3: T difference (final - IC) ──
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5))

dT = T_final[k_surf, :, :] - T_ic[k_surf, :, :]
im = axes3[0].pcolormesh(lon, lat, dT, cmap="RdBu_r", shading="auto", vmin=-5, vmax=5)
axes3[0].set_title("T(final) - T(IC) at surface (K)")
axes3[0].set_ylabel("Latitude")
axes3[0].set_xlabel("Longitude")
plt.colorbar(im, ax=axes3[0])

dT_mid = T_final[k_mid, :, :] - T_ic[k_mid, :, :]
im = axes3[1].pcolormesh(lon, lat, dT_mid, cmap="RdBu_r", shading="auto", vmin=-5, vmax=5)
axes3[1].set_title(f"T(final) - T(IC) at z={z[k_mid]:.0f}m (K)")
axes3[1].set_xlabel("Longitude")
plt.colorbar(im, ax=axes3[1])

fig3.suptitle("Temperature change over 3.1h simulation", fontsize=14)
plt.tight_layout()
plt.savefig("final_state_diff.png", dpi=150, bbox_inches="tight")
print("Saved final_state_diff.png")

print("All plots saved.")
