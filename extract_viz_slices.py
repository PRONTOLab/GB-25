"""Extract 6 2D slices from the latest 1/24° continued checkpoint into a
small JLD2/HDF5 file for remote visualization.

Usage: ITER=9000 python3 extract_viz_slices.py
"""
import os, sys
import numpy as np
import h5py

iter_num = int(os.environ.get("ITER", "9000"))
src = f"/teamspace/studios/this_studio/GB-25/simulations/initial_conditions/twentyfourth_continued_iter{iter_num}_assembled.jld2"
dst = f"/teamspace/studios/this_studio/GB-25/viz_slices_iter{iter_num}.h5"

if not os.path.isfile(src):
    print(f"ERROR: missing {src}", file=sys.stderr); sys.exit(1)

k_mid = 15      # ~7 km (Δz = 30000/64 = 468.75 m → z(k=15) ≈ 7.27 km)
Δz = 30000.0 / 64.0  # m

print(f"Reading {src}")
with h5py.File(src, "r") as f:
    # h5py sees Julia's column-major as reversed: shape (Nz, Ny, Nx)
    # So f["ρ"][0] gives the k=0 (surface) slice with shape (Ny, Nx).
    # We transpose each slice to (Nx, Ny) for Julia-natural row-major analysis.
    Nz, Ny, Nx = f["ρ"].shape
    print(f"  grid: Nx={Nx}, Ny={Ny}, Nz={Nz}")

    rho_sfc  = np.array(f["ρ"][0]).T                 # (Nx, Ny)
    rho_mid  = np.array(f["ρ"][k_mid]).T

    # Momentum at surface
    ru_sfc   = np.array(f["ρu"][0]).T                # Periodic x-Face, same (Nx, Ny)
    # ρv is (Nz, Ny+1, Nx) — trim the bottom boundary row so we have (Nx, Ny)
    # h5py sees it as (Nz, Ny+1, Nx), taking [0, :Ny, :] gives (Ny, Nx)
    rv_sfc_full = np.array(f["ρv"][0])               # (Ny+1, Nx)
    rv_sfc   = rv_sfc_full[:Ny, :].T                 # (Nx, Ny)

    # Moisture
    rqv_sfc  = np.array(f["ρqᵛ"][0]).T               # (Nx, Ny) — centers
    rqcl_mid = np.array(f["micro_ρqᶜˡ"][k_mid]).T    # (Nx, Ny)

    # ρw has Nz+1 z-face levels; interior k-index for mid-level = k_mid
    # f["ρw"].shape = (Nz+1, Ny, Nx)
    rw_mid   = np.array(f["ρw"][k_mid]).T            # (Nx, Ny)

    # Vertically integrated rain: Σ ρqʳ * Δz  →  kg/m² (column mass)
    # Read entire ρqʳ volume — 8640×3840×64 Float32 = 8.5 GB
    print("  loading ρqʳ full volume for column integration…")
    rqr = np.array(f["ρqʳ"])                          # (Nz, Ny, Nx)
    qr_vertint = (rqr.sum(axis=0) * Δz).T             # (Nx, Ny), kg/m²
    del rqr

# Derived (per-mass) fields
safe_sfc = np.where(rho_sfc > 0.001, rho_sfc, 0.001)
safe_mid = np.where(rho_mid > 0.001, rho_mid, 0.001)

u_sfc   = ru_sfc  / safe_sfc
v_sfc   = rv_sfc  / safe_sfc
qv_sfc  = rqv_sfc / safe_sfc * 1000.0                 # g/kg
qcl_mid = rqcl_mid / safe_mid * 1000.0                # g/kg
w_mid   = rw_mid  / safe_mid                           # m/s

lat = np.linspace(-80.0, 80.0, Ny, endpoint=False, dtype=np.float32)
lon = np.linspace(0.0, 360.0, Nx, endpoint=False, dtype=np.float32)

print(f"Writing {dst}")
with h5py.File(dst, "w") as f:
    f.attrs["iteration"] = iter_num
    f.attrs["sim_time_h"] = iter_num * 0.8 / 3600.0
    f.attrs["k_mid"] = k_mid
    f.attrs["z_mid_km"] = (k_mid + 0.5) * Δz / 1000.0
    f.attrs["dz_m"] = Δz
    f.attrs["source"] = os.path.basename(src)
    f.create_dataset("lon_deg", data=lon, compression="gzip", compression_opts=4)
    f.create_dataset("lat_deg", data=lat, compression="gzip", compression_opts=4)
    f.create_dataset("u_sfc",   data=u_sfc.astype(np.float32),   compression="gzip", compression_opts=4)
    f.create_dataset("v_sfc",   data=v_sfc.astype(np.float32),   compression="gzip", compression_opts=4)
    f.create_dataset("qv_sfc",  data=qv_sfc.astype(np.float32),  compression="gzip", compression_opts=4)
    f.create_dataset("qr_vertint", data=qr_vertint.astype(np.float32), compression="gzip", compression_opts=4)
    f.create_dataset("qcl_mid", data=qcl_mid.astype(np.float32), compression="gzip", compression_opts=4)
    f.create_dataset("w_mid",   data=w_mid.astype(np.float32),   compression="gzip", compression_opts=4)

print(f"  u_sfc   range: [{u_sfc.min():.2f}, {u_sfc.max():.2f}] m/s")
print(f"  v_sfc   range: [{v_sfc.min():.2f}, {v_sfc.max():.2f}] m/s")
print(f"  qv_sfc  range: [{qv_sfc.min():.3f}, {qv_sfc.max():.3f}] g/kg")
print(f"  qr_vertint range: [{qr_vertint.min():.4f}, {qr_vertint.max():.4f}] kg/m²")
print(f"  qcl_mid range: [{qcl_mid.min():.3f}, {qcl_mid.max():.3f}] g/kg")
print(f"  w_mid   range: [{w_mid.min():.2f}, {w_mid.max():.2f}] m/s")

sz = os.path.getsize(dst) / 1e6
print(f"Saved {dst} ({sz:.1f} MB)")
