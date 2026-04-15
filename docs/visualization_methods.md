# Visualization methods for the 1/24° cascade

Concise reference for how each figure type is produced and what scripts exist.
All scripts run from the repo root (`/teamspace/studios/this_studio/GB-25`).

## Data sources

Two tiers of data, chosen by what the viz actually needs:

| Source | Contents | Size | Used by |
|---|---|---|---|
| `simulations/output/nccl_8gpu_24th_deg_continued/fields_rank{0..7}_iter{NNN}.jld2` | Per-rank JLD2 interiors | ~19 GB/rank × 8 = 152 GB/save | fast 2D python plots |
| `simulations/initial_conditions/twentyfourth_continued_iter{N}_assembled.jld2` | Single-file global assembled | ~80 GB/save | sphere viz (needs whole grid), small-slice extraction |
| `viz_slices_iter{N}.h5` | 6 precomputed 2D slices at Float32 with gzip | ~700 MB | remote / low-spec rendering |

The assembled files are written by `assemble_24th_continued.jl` (param: `ITER=NNNN`).
The small slice bundles are written by `extract_viz_slices.py` (param: `ITER=NNNN`).

## Rank → global tile mapping (repeated everywhere)

For `Partition(Rx=4, Ry=2, 1)` with 8 NCCL ranks:
```
ix = rank ÷ Ry        # 0..Rx-1 (longitude tile index)
iy = rank % Ry        # 0..Ry-1 (latitude tile index)
x0 = ix * Nx_per      # Nx_per = 2160 at 1/24°
y0 = iy * Ny_per      # Ny_per = 1920
```
Face-field boundary row on `ρv` lives only on top-y rank (`iy == Ry-1`).
Face row on `ρw` lives on every rank (z is not partitioned).

## Flat 2D visualizations (fast, Python + h5py)

Script: `plot_24th_continued.py` — surface + mid-level panels.

- Reads only rank tiles from per-rank JLD2 files; no assembly needed.
- Surface (k=0) for centers/ρu; mid-level (k=15 ≈ 7 km) for w and qci.
- ρv trimmed to center-size for plotting (drops the wall face row on top rank).
- Subsampled `ss=4` for speed; output ~5 MB PNG at 150 dpi.
- Emits a HEALTH line (NaN/finite check, ρ-min, field ranges) for monitoring.
- Uses explicit `(vmin, vmax)` for every panel so saturating colours reveal extremes.

Usage: `ITER=9000 python3 plot_24th_continued.py` → `continued_iter9000_surface.png`.

## Sphere visualizations (Oceananigans + CairoMakie)

Two scripts, both read from the **assembled** JLD2:

1. `visualize_sphere_24th_continued.jl` — 3-panel: θ, qv, wind speed. Uses `ρu` for approximate `|u|` (ρv assembled has `Nφ+1` rows, awkward to drop in-Julia).
2. `visualize_sphere_qv_qci.jl` — 2-panel: surface qv + mid-level qci. Tight layout (negative `colgap!`, `protrusions=0`) for closer spheres.

Both use:
```julia
grid = LatitudeLongitudeGrid(CPU(); size=(Nλ,Nφ,Nz), halo=(4,4,4),
                             latitude=(-80,80), longitude=(0,360), z=(0,30e3))
field = CenterField(grid)
Oceananigans.interior(field)[:, :, 1] .= data_2D
Axis3(...); surface!(ax, view(field, :, :, 1); colormap=…, colorrange=…)
save(path, fig; px_per_unit=3)  # or 4 for final
```
`Axis3 + surface!` with a LatitudeLongitudeGrid Field automatically projects to the
3D sphere via the Oceananigans Makie extension.

**Render cost** (empirical, CPU-only CairoMakie, 1 thread):
- 1/16° (5760×2560), 3 panels, px_per_unit=4: ~5 min
- 1/24° (8640×3840), 3 panels, px_per_unit=4: **~22 min**, 138 GB peak RSS
- 1/24° (8640×3840), 2 panels, px_per_unit=3: ~8 min

For faster iteration, drop to `px_per_unit=2` (still crisp at 1/24°) or render at a
subsampled grid.

## Remote-friendly slice bundle

For rendering on another machine or in a notebook without 80 GB of RAM:

Script: `extract_viz_slices.jl` (native JLD2; `.py` variant exists but JLD2 is preferred).
Produces `viz_slices_iter{N}.jld2` with 6 fields + coords:

| Dataset | Units | Shape |
|---|---|---|
| `lon_deg`, `lat_deg` | ° | (8640,), (3840,) |
| `u_sfc`, `v_sfc` | m/s | (8640, 3840) |
| `qv_sfc` | g/kg | (8640, 3840) |
| `qr_vertint` | kg/m² (column mass) | (8640, 3840) |
| `qcl_mid` (k=15, ~7 km) | g/kg | (8640, 3840) |
| `w_mid` (k=15) | m/s | (8640, 3840) |

Top-level keys: `iteration`, `sim_time_h`, `k_mid`, `z_mid_km`, `dz_m`, `source`.
Float32 arrays → ~760 MB.

### Loading in Julia (preferred)
```julia
using JLD2
f = JLD2.jldopen("viz_slices_iter9000.jld2", "r")
qv   = f["qv_sfc"]             # (8640, 3840), Float32
lon  = f["lon_deg"]
lat  = f["lat_deg"]
hours = f["sim_time_h"]
close(f)
```

### Loading in Python
```python
import h5py  # JLD2 writes HDF5 under the hood; works for simple array payloads
with h5py.File("viz_slices_iter9000.jld2", "r") as f:
    qv = f["qv_sfc"][:]         # (3840, 8640) in NumPy row-major; transpose if needed
```

## Diagnostics from the simulation log

`plot_diagnostics.py` parses the per-rank diag lines the simulation emits every
100 iterations:
```
[ Info: [r0] iter 9000 t=7200.0s Δt=0.80 wall=10146.2s ρ=[…] ρw=[…]
```
Produces time-series plots of ρ-min/max and ρw-min/max per rank — useful for
catching blowups.

## Which script to use, when

| Need | Script |
|---|---|
| Fast sanity-check panel after a save | `plot_24th_continued.py` |
| High-res sphere for a paper/slides | `visualize_sphere_24th_continued.jl` |
| 2-panel sphere with more detail per panel | `visualize_sphere_qv_qci.jl` |
| Transfer fields to another machine for custom rendering | `extract_viz_slices.py` |
| Time-series of simulation health | `plot_diagnostics.py` |
