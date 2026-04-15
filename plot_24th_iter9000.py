"""Surface viz of latest 1/24° checkpoint (iter 9000, t=2h)."""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

base = "/teamspace/studios/this_studio/GB-25/simulations/output/nccl_8gpu_24th_deg"
iter_str = "009000"

Rx, Ry = 4, 2
Nx_per, Ny_per = 2160, 1920
Nλ, Nφ, Nz = Rx * Nx_per, Ry * Ny_per, 64
print(f"Global: {Nλ} × {Nφ} × {Nz}")

# Surface (k=0) for centers + ρu/ρv; mid-level (k=32) for ρw and ρqᶜⁱ to see clouds/updrafts aloft
k_mid = 15  # ~7 km altitude with Δz=469 m (mid-troposphere)

center_surface = ["ρ", "ρu", "ρθ", "ρqᵛ", "ρqᶜˡ", "ρqʳ"]
fields = {n: np.zeros((Nλ, Nφ), dtype=np.float32) for n in center_surface}
fields["ρv"] = np.zeros((Nλ, Nφ), dtype=np.float32)  # surface, trim top face on iy=1
fields["ρw_mid"] = np.zeros((Nλ, Nφ), dtype=np.float32)  # mid-level
fields["ρqᶜⁱ_mid"] = np.zeros((Nλ, Nφ), dtype=np.float32)  # mid-level
fields["ρ_mid"] = np.zeros((Nλ, Nφ), dtype=np.float32)  # for normalizing mid-level

for r in range(8):
    ix, iy = r // Ry, r % Ry
    path = f"{base}/fields_rank{r}_iter{iter_str}.jld2"
    print(f"  rank {r} (ix={ix} iy={iy}) ← {os.path.basename(path)}")
    f = h5py.File(path, "r")
    x0, y0 = ix * Nx_per, iy * Ny_per
    for n in center_surface:
        slc = np.array(f[n][0]).T  # (Nx, Ny) — surface
        fields[n][x0:x0 + Nx_per, y0:y0 + Ny_per] = slc
    rv = np.array(f["ρv"][0]).T
    fields["ρv"][x0:x0 + Nx_per, y0:y0 + Ny_per] = rv[:, :Ny_per]
    # mid-level (k=32)
    fields["ρw_mid"][x0:x0 + Nx_per, y0:y0 + Ny_per] = np.array(f["ρw"][k_mid]).T
    fields["ρqᶜⁱ_mid"][x0:x0 + Nx_per, y0:y0 + Ny_per] = np.array(f["ρqᶜⁱ"][k_mid]).T
    fields["ρ_mid"][x0:x0 + Nx_per, y0:y0 + Ny_per] = np.array(f["ρ"][k_mid]).T
    f.close()

print("Stitched. Computing derived fields…")
rho = fields["ρ"]
safe = np.where(rho > 0.001, rho, 0.001)
theta = fields["ρθ"] / safe
qv = fields["ρqᵛ"] / safe * 1000  # g/kg
u = fields["ρu"] / safe
v = fields["ρv"] / safe
wspd = np.sqrt(u**2 + v**2)
qcl = fields["ρqᶜˡ"] / safe * 1000
qr = fields["ρqʳ"] / safe * 1000

rho_mid = fields["ρ_mid"]
safe_mid = np.where(rho_mid > 0.001, rho_mid, 0.001)
w_mid = fields["ρw_mid"] / safe_mid
qci_mid = fields["ρqᶜⁱ_mid"] / safe_mid * 1000

print("Plotting…")
lon = np.linspace(0, 360, Nλ, endpoint=False)
lat = np.linspace(-80, 80, Nφ, endpoint=False)

fig, axes = plt.subplots(3, 3, figsize=(24, 14))
fig.suptitle(f"1/24° (Nλ={Nλ}, Nφ={Nφ}) — iter 9000, t=2h sim — surface k=0", fontsize=14)

cfgs = [
    ("θ [K]",         theta,    "RdYlBu_r", (250, 320)),
    ("qv [g/kg]",     qv,       "YlGnBu",   (0, 34)),
    ("wind speed [m/s]", wspd,  "hot_r",    (0, 35)),
    ("u [m/s]",       u,        "RdBu_r",   (-30, 30)),
    ("v [m/s]",       v,        "RdBu_r",   (-30, 30)),
    (f"w [m/s] (k={k_mid}, ~7km)", w_mid, "RdBu_r", (-2, 2)),
    ("qcl [g/kg] (sfc)",  qcl,  "Blues",    (0, 1)),
    (f"qci [g/kg] (k={k_mid}, ~7km)", qci_mid, "Purples", (0, 10)),
    ("qr [g/kg] (sfc)",   qr,   "Oranges",  (0, 1)),
]

# subsample for speed (8640/4 = 2160 plot resolution, fine)
ss = 4
for idx, (name, data, cmap, clim) in enumerate(cfgs):
    ax = axes[idx // 3, idx % 3]
    d = data[::ss, ::ss].T  # (Ny, Nx)
    im = ax.pcolormesh(lon[::ss], lat[::ss], d,
                        cmap=cmap, vmin=clim[0], vmax=clim[1],
                        shading="nearest", rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(name)
    if idx % 3 == 0: ax.set_ylabel("lat [°]")
    if idx >= 6:     ax.set_xlabel("lon [°]")

plt.tight_layout()
out = "twentyfourth_iter9000_surface.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
