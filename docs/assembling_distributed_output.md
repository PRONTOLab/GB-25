# Assembling per-rank JLD2 output from NCCLDistributed runs

## Problem

Oceananigans' `JLD2Writer` has a serialization bug with `NCCLDistributed`
grids (the grid contains non-serializable NCCL communicators and MPI
handles). The workaround is a custom per-rank output callback that writes
one JLD2 file per rank per snapshot, e.g.:

```
fields_rank0_iter007500.jld2
fields_rank1_iter007500.jld2
...
fields_rank7_iter007500.jld2
```

Each file contains the **interior** (no halos) of each prognostic field
for that rank's subdomain tile.

## Reading with Python (h5py)

JLD2 is HDF5-based, so `h5py` reads it directly. However, Julia stores
arrays in **column-major** (Fortran) order while HDF5/h5py returns them
in **row-major** (C) order. A Julia array of shape `(Nx, Ny, Nz)` is
read by h5py as shape `(Nz, Ny, Nx)`.

**Fix**: transpose after reading:

```python
import h5py
import numpy as np

with h5py.File("fields_rank0_iter007500.jld2", "r") as f:
    rho = np.array(f["ρ"]).T  # (Nz, Ny, Nx) -> (Nx, Ny, Nz)
```

## Rank-to-tile mapping

For `Partition(Rx, Ry)` (e.g. `Rx=4, Ry=2` for 8 GPUs), Oceananigans
maps MPI ranks to subdomain tiles as:

```
ix = rank ÷ Ry      (longitude index, 0-based)
iy = rank % Ry      (latitude index, 0-based)
```

So for `Rx=4, Ry=2`:

| rank | ix | iy | longitude band | latitude band |
|------|----|----|----------------|---------------|
| 0 | 0 | 0 | 0°–90° | south (-80°–0°) |
| 1 | 0 | 1 | 0°–90° | north (0°–80°) |
| 2 | 1 | 0 | 90°–180° | south |
| 3 | 1 | 1 | 90°–180° | north |
| 4 | 2 | 0 | 180°–270° | south |
| 5 | 2 | 1 | 180°–270° | north |
| 6 | 3 | 0 | 270°–360° | south |
| 7 | 3 | 1 | 270°–360° | north |

## Assembly code (Python)

```python
import numpy as np
import h5py
import os

Rx, Ry = 4, 2
output_dir = "simulations/output/nccl_8gpu_24th_deg"
iter_str = "015000"

def load_field(name):
    tiles = []
    for r in range(Rx * Ry):
        path = os.path.join(output_dir, f"fields_rank{r}_iter{iter_str}.jld2")
        with h5py.File(path, "r") as f:
            tiles.append(np.array(f[name]).T)  # transpose for Julia→C order

    nx, ny, nz = tiles[0].shape

    global_field = np.zeros((Rx * nx, Ry * ny, nz), dtype=tiles[0].dtype)
    for r in range(Rx * Ry):
        ix = r // Ry
        iy = r % Ry
        global_field[ix*nx:(ix+1)*nx, iy*ny:(iy+1)*ny, :] = tiles[r]

    return global_field

# Example: load density and plot a horizontal slice
rho = load_field("ρ")
print(f"Global shape: {rho.shape}")  # (8640, 3840, 64)

# Horizontal slice at surface
import matplotlib.pyplot as plt
lon = np.linspace(0, 360, rho.shape[0], endpoint=False)
lat = np.linspace(-80, 80, rho.shape[1], endpoint=False)
plt.pcolormesh(lon, lat, rho[:, :, 0].T, shading='nearest')
plt.colorbar(label='ρ [kg/m³]')
plt.xlabel('lon [°]'); plt.ylabel('lat [°]')
plt.title('Surface density')
plt.savefig('surface_rho.png', dpi=150)
```

## Available fields in each JLD2 file

| Key | Description | Location |
|-----|-------------|----------|
| `ρ` | density | Center |
| `ρu` | x-momentum density | XFace |
| `ρv` | y-momentum density | YFace |
| `ρw` | z-momentum density | ZFace |
| `ρθ` | potential temperature density | Center |
| `ρqᵛ` | water vapor density | Center |
| `ρqᶜˡ` | cloud liquid density | Center |
| `ρqᶜⁱ` | cloud ice density | Center |
| `ρqʳ` | rain density | Center |
| `ρqˢ` | snow density | Center |
| `iteration` | iteration number | scalar |
| `time` | simulation time [s] | scalar |
| `Δt` | time step [s] | scalar |

## Notes

- Face fields (ρu, ρv, ρw) have one extra element in their face
  direction compared to center fields on bounded dimensions. For
  approximate visualization, treating them as center-located is fine.
- The interior data excludes halos, so tiles abut exactly with no overlap.
- Each per-rank file is ~19 GB (Float32, 10 fields × ~2 GB each).
