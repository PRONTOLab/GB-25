#!/usr/bin/env python3
"""Plot extracted 2D slices from a GB-25 checkpoint.

Usage: plot_checkpoint.py <viz_dir> [<output_png_dir>]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

viz_dir = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else viz_dir
os.makedirs(out_dir, exist_ok=True)

# Parse manifest
m = {}
with open(os.path.join(viz_dir, "manifest.txt")) as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        m[k.strip()] = v.strip()
Nx = int(m["Nx"]); Ny = int(m["Ny"])
print(f"Grid {Nx}x{Ny}  from {m.get('checkpoint')}")

def load(name):
    path = os.path.join(viz_dir, f"{name}.f32")
    a = np.fromfile(path, dtype=np.float32)
    # Julia wrote column-major (Nx, Ny) → reshape fortran-order
    return a.reshape((Nx, Ny), order="F").T  # transpose so row=lat, col=lon

# Lat/lon coord (approximate, full sphere baroclinic wave domain — tweak if needed)
lon = np.linspace(0, 360, Nx, endpoint=False)
lat = np.linspace(-80, 80, Ny)  # placeholder; real grid in Oceananigans

def show(name, cmap="viridis", pct=(1, 99), title=None, unit=""):
    A = load(name)
    vmin, vmax = np.percentile(A, pct)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(A, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_title(title or name)
    ax.set_xlabel("lon (°)"); ax.set_ylabel("lat (°)")
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(unit)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}  (min={A.min():.3e} max={A.max():.3e} mean={A.mean():.3e})")
    return A

# Surface maps
show("surf_T",  cmap="RdBu_r", title="Surface T (interior k=1)",          unit="K")
show("surf_u",  cmap="RdBu_r", title="Surface u",                          unit="m/s")
show("surf_v",  cmap="RdBu_r", title="Surface v",                          unit="m/s")
show("surf_w",  cmap="RdBu_r", title="Surface w",                          unit="m/s")
show("surf_qᵛ", cmap="Blues",  title="Surface water vapor qᵛ",             unit="kg/kg")
show("surf_ρ",  cmap="viridis",title="Surface density ρ",                  unit="kg/m³")
show("surf_θ",  cmap="RdBu_r", title="Surface potential temp θ",           unit="K")

# Mid-level
show("mid_T",  cmap="RdBu_r", title="Mid-level (k=32) T",     unit="K")
show("mid_u",  cmap="RdBu_r", title="Mid-level u",            unit="m/s")
show("mid_v",  cmap="RdBu_r", title="Mid-level v",            unit="m/s")
show("mid_w",  cmap="RdBu_r", title="Mid-level w",            unit="m/s")
show("mid_qᵛ", cmap="Blues",  title="Mid-level qᵛ",           unit="kg/kg")

# Column-integrated cloud / precip (proxy — missing proper density/dz weighting)
show("cloud_path", cmap="Greys_r", pct=(50, 99.5),
     title="Sum over z of (qˡ + qᶜˡ + qⁱ + qᶜⁱ) — cloud water proxy",
     unit="kg/kg (summed)")
show("rain_path",  cmap="Purples", pct=(50, 99.5),
     title="Sum over z of (qʳ + qˢ) — precip proxy",
     unit="kg/kg (summed)")

# Combined quicklook panel
fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
def panel(ax, name, cmap, title):
    A = load(name)
    vmin, vmax = np.percentile(A, (1, 99))
    im = ax.imshow(A, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("lon (°)"); ax.set_ylabel("lat (°)")
    plt.colorbar(im, ax=ax, shrink=0.85)
panel(axes[0,0], "surf_T",    "RdBu_r", "surface T")
panel(axes[0,1], "surf_qᵛ",   "Blues",  "surface qᵛ")
panel(axes[0,2], "cloud_path","Greys_r","cloud water path (proxy)")
panel(axes[1,0], "mid_w",     "RdBu_r", "mid-level w")
panel(axes[1,1], "mid_T",     "RdBu_r", "mid-level T")
panel(axes[1,2], "rain_path", "Purples","precip path (proxy)")
out = os.path.join(out_dir, "quicklook.png")
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"wrote {out}")
