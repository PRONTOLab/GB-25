"""Surface viz for a 1/24° continuation checkpoint. Usage: ITER=4500 python3 plot_24th_continued.py"""
import numpy as np
import matplotlib.pyplot as plt
import h5py, os, sys

iter_num = int(os.environ.get("ITER", "4500"))
iter_str = f"{iter_num:06d}"
base = "/teamspace/studios/this_studio/GB-25/simulations/output/nccl_8gpu_24th_deg_continued"

Rx, Ry = 4, 2
Nx_per, Ny_per = 2160, 1920
Nλ, Nφ, Nz = Rx * Nx_per, Ry * Ny_per, 64
k_mid = 15  # ~7 km
print(f"Global: {Nλ} × {Nφ} × {Nz}, iter {iter_num}")

center_surface = ["ρ", "ρu", "ρθ", "ρqᵛ", "ρqᶜˡ", "ρqʳ"]
fields = {n: np.zeros((Nλ, Nφ), dtype=np.float32) for n in center_surface}
fields["ρv"] = np.zeros((Nλ, Nφ), dtype=np.float32)
fields["ρw_mid"] = np.zeros((Nλ, Nφ), dtype=np.float32)
fields["ρqᶜⁱ_mid"] = np.zeros((Nλ, Nφ), dtype=np.float32)
fields["ρ_mid"] = np.zeros((Nλ, Nφ), dtype=np.float32)

has_nan = False
for r in range(8):
    ix, iy = r // Ry, r % Ry
    path = f"{base}/fields_rank{r}_iter{iter_str}.jld2"
    if not os.path.isfile(path):
        print(f"  MISSING: {path}", file=sys.stderr); sys.exit(1)
    f = h5py.File(path, "r")
    x0, y0 = ix * Nx_per, iy * Ny_per
    for n in center_surface:
        slc = np.array(f[n][0]).T
        if not np.all(np.isfinite(slc)):
            has_nan = True
            print(f"  NaN/Inf in {n} rank {r}", file=sys.stderr)
        fields[n][x0:x0 + Nx_per, y0:y0 + Ny_per] = slc
    rv = np.array(f["ρv"][0]).T
    fields["ρv"][x0:x0 + Nx_per, y0:y0 + Ny_per] = rv[:, :Ny_per]
    fields["ρw_mid"][x0:x0 + Nx_per, y0:y0 + Ny_per] = np.array(f["ρw"][k_mid]).T
    fields["ρqᶜⁱ_mid"][x0:x0 + Nx_per, y0:y0 + Ny_per] = np.array(f["ρqᶜⁱ"][k_mid]).T
    fields["ρ_mid"][x0:x0 + Nx_per, y0:y0 + Ny_per] = np.array(f["ρ"][k_mid]).T
    f.close()

if has_nan:
    print("HEALTH: NaN/Inf detected")
else:
    print("HEALTH: all finite ✓")

rho = fields["ρ"]
print(f"  ρ   surface: [{rho.min():.4f}, {rho.max():.4f}]  zeros={(rho <= 0).sum()}")
print(f"  ρw  k={k_mid}: [{fields['ρw_mid'].min():.2f}, {fields['ρw_mid'].max():.2f}]")

safe = np.where(rho > 0.001, rho, 0.001)
theta = fields["ρθ"] / safe
qv = fields["ρqᵛ"] / safe * 1000
u = fields["ρu"] / safe
v = fields["ρv"] / safe
wspd = np.sqrt(u**2 + v**2)
qcl = fields["ρqᶜˡ"] / safe * 1000
qr = fields["ρqʳ"] / safe * 1000
safe_mid = np.where(fields["ρ_mid"] > 0.001, fields["ρ_mid"], 0.001)
w_mid = fields["ρw_mid"] / safe_mid
qci_mid = fields["ρqᶜⁱ_mid"] / safe_mid * 1000

print(f"  θ   surface: [{theta.min():.1f}, {theta.max():.1f}]  qv surface: [{qv.min():.2f}, {qv.max():.2f}]")

lon = np.linspace(0, 360, Nλ, endpoint=False)
lat = np.linspace(-80, 80, Nφ, endpoint=False)

fig, axes = plt.subplots(3, 3, figsize=(24, 14))
fig.suptitle(f"1/24° continuation — iter {iter_num}, t={iter_num*0.8/3600:.2f}h sim (surface k=0)", fontsize=14)

cfgs = [
    ("θ [K]",            theta,   "RdYlBu_r", (250, 320)),
    ("qv [g/kg]",        qv,      "YlGnBu",   (0, 34)),
    ("wind speed [m/s]", wspd,    "hot_r",    (0, 35)),
    ("u [m/s]",          u,       "RdBu_r",   (-30, 30)),
    ("v [m/s]",          v,       "RdBu_r",   (-30, 30)),
    (f"w [m/s] (k={k_mid}, ~7km)", w_mid,     "RdBu_r",  (-2, 2)),
    ("qcl [g/kg] (sfc)", qcl,     "Blues",    (0, 1)),
    (f"qci [g/kg] (k={k_mid}, ~7km)", qci_mid, "Purples", (0, 10)),
    ("qr [g/kg] (sfc)",  qr,      "Oranges",  (0, 1)),
]
ss = 4
for idx, (name, data, cmap, clim) in enumerate(cfgs):
    ax = axes[idx // 3, idx % 3]
    d = data[::ss, ::ss].T
    im = ax.pcolormesh(lon[::ss], lat[::ss], d, cmap=cmap, vmin=clim[0], vmax=clim[1],
                        shading="nearest", rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(name)
    if idx % 3 == 0: ax.set_ylabel("lat [°]")
    if idx >= 6:     ax.set_xlabel("lon [°]")

plt.tight_layout()
out = f"continued_iter{iter_num}_surface.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
