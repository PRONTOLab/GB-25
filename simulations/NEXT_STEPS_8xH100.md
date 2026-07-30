# Higher-resolution ocean simulation on 8x H100

Planning notes for continuing the ocean spinup at 1/24 degree on 8 H100 GPUs.

## Starter script

`simulations/ocean_24th_degree_8gpu.jl` is a working starter script that picks
up from the 1/10 degree checkpoint produced by `ocean_tenth_degree_from_quarter.jl`.

## Checkpoint input

The 1/10 degree checkpoint (u, v, T, S at 3840 x 1920 x 64, after 180 days of
1/4 spinup + 15 days of 1/10 refinement, jet at ±35 degrees) is on Dropbox:

`/ocean_tenth_degree_15day_jet35.jld2` (15 GB)

Download with `dbxcli get` or via the Dropbox UI, place in
`simulations/checkpoints/` and pass its path as the first positional argument
to the script, or let the script auto-detect the most recent
`ocean_tenth_degree_15day_*.jld2` in that directory.

## Hardware target

- 8x NVIDIA H100 (80 GB each) = 640 GB total VRAM
- Multi-GPU via Oceananigans.Distributed + MPI (NOT Reactant -- Reactant poisons CUDA kernels)
- See `SIMULATION_NOTES.md` for the LD_LIBRARY_PATH cleanup and the
  "don't mix Reactant and Oceananigans GPU" rule.

## Target resolution: 1/24 degree

Grid: 9216 x 4608 x 64

Memory per 3D field = (Nx+16) x (Ny+16) x (Nz+16) x 8 bytes (Float64, halo=8).
Estimate ~18 3D fields for HydrostaticFreeSurfaceModel with T, S, no closure.

| Resolution | Nx x Ny x Nz    | Per field | ~18 fields | Per GPU (8-way) | Fits? |
|------------|-----------------|-----------|------------|-----------------|-------|
| 1/10       | 3840x1920x64    | 4.8 GB    | 86 GB      | 10.8 GB         | Easy  |
| 1/20       | 7680x3840x64    | 19 GB     | 342 GB     | 43 GB           | Yes   |
| **1/24**   | **9216x4608x64**| **25 GB** | **457 GB** | **57 GB**       | **Yes** |
| 1/25       | 9600x4800x64    | 29.5 GB   | 531 GB     | 66 GB           | Tight |
| 1/32       | 12288x6144x64   | 48 GB     | 864 GB     | 108 GB          | No    |

**1/24 degree** leaves ~23 GB headroom per GPU for WENO temporaries and CUDA overhead.

## CFL and time step

Based on the 1/4 degree spinup, U_max ~ 12 m/s at equilibrium.

dx_min = (360/Nx) * cos(80 deg) * 111000 m

| Resolution | dx_min (80 lat) | dt for CFL=0.2 | Barotropic CFL (60 substeps) |
|------------|-----------------|-----------------|------------------------------|
| 1/10       | 1829 m          | 30 s            | 0.054                        |
| 1/20       | 915 m           | 15 s            | 0.054                        |
| **1/24**   | **753 m**       | **12 s**        | **0.053**                    |
| 1/25       | 732 m           | 12 s            | 0.054                        |

**Use Dt = 12 seconds** with substeps=60 for 1/24 degree.

## GPU partitioning

For 8 GPUs with 1/24 degree (9216 x 4608):

| Partition | Per-GPU Nx x Ny | Shape   | Notes                          |
|-----------|-----------------|---------|--------------------------------|
| **4x2**   | **2304 x 2304** | Square  | **Best -- perfectly balanced** |
| 2x4       | 4608 x 1152     | 4:1     | OK but elongated               |
| 8x1       | 1152 x 4608     | 1:4     | Avoid                          |

**Partition(4, 2, 1)** gives perfectly square 2304x2304 subdomains per GPU.

## Script structure

```julia
using MPI
MPI.Init()

using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials
using CUDA
using JLD2

const Nx = 9216   # 1/24 degree
const Ny = 4608
const Nz = 64
const Dt = 12     # seconds; CFL ~ 0.2 at U_max = 12 m/s, dx_min = 753 m
const total_time = 15days
const Ninner = 64
const Nouter = ceil(Int, total_time / (Ninner * Dt))

arch = Oceananigans.Distributed(GPU(); partition=Partition(4, 2, 1))
rank = MPI.Comm_rank(MPI.COMM_WORLD)

grid = LatitudeLongitudeGrid(arch;
    size = (Nx, Ny, Nz),
    halo = (8, 8, 8),
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)

model = HydrostaticFreeSurfaceModel(;
    grid,
    free_surface = SplitExplicitFreeSurface(substeps=60),
    buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState(Float64)),
    closure = nothing,
    coriolis = HydrostaticSphericalCoriolis(),
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    tracers = (:T, :S),
)

# --- Load and interpolate from 1/10 degree checkpoint ---
# See "IC interpolation" section below

model.clock.last_Dt = Dt

# --- Time stepping ---
Oceananigans.TimeSteppers.first_time_step!(model, Dt)

for k in 1:Nouter
    for _ in 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Dt)
    end
    CUDA.synchronize()
    # ... logging, CFL check ...
end

# --- Save checkpoint (rank 0 gathers or each rank saves its partition) ---
```

## IC interpolation from 1/10 checkpoint

Key gotchas (from SIMULATION_NOTES.md):
1. Source grid must be on **GPU**, not CPU -- `interpolate!` fails cross-architecture
2. Use `set!(src_field, FT.(data))` to upload host arrays to device
3. For distributed targets, each rank needs to interpolate onto its local partition

Strategy for distributed interpolation:
- **Option A (simple)**: Every rank loads the full 1/10 checkpoint onto its GPU,
  creates a full-size source grid, and Oceananigans.Fields.interpolate! handles
  the local partition automatically. The 1/10 source is ~5 GB per field (4 fields
  = 20 GB) -- fits alongside the 1/24 target fields on each 80 GB GPU.
- **Option B (memory-efficient)**: Pre-partition source data. Not needed here since
  source is small relative to GPU memory.

```julia
# Each rank loads the full checkpoint
checkpoint_path = "simulations/checkpoints/ocean_tenth_degree_15day_XXXX.jld2"
Nx_src, Ny_src, Nz_src, u_data, v_data, T_data, S_data = JLD2.jldopen(checkpoint_path, "r") do f
    (f["Nx"], f["Ny"], f["Nz"], f["u"], f["v"], f["T"], f["S"])
end

# Source grid on GPU (must match target architecture)
source_grid = LatitudeLongitudeGrid(GPU();
    size = (Nx_src, Ny_src, Nz_src),
    halo = (1, 1, 1),
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)

FT = Float64
T_src = CenterField(source_grid); set!(T_src, FT.(T_data))
S_src = CenterField(source_grid); set!(S_src, FT.(S_data))
u_src = XFaceField(source_grid);  set!(u_src, FT.(u_data))
v_src = YFaceField(source_grid);  set!(v_src, FT.(v_data))

for (src, dst) in [(T_src, model.tracers.T), (S_src, model.tracers.S),
                    (u_src, model.velocities.u), (v_src, model.velocities.v)]
    Oceananigans.BoundaryConditions.fill_halo_regions!(src)
    Oceananigans.Fields.interpolate!(dst, src)
end
```

**IMPORTANT**: For distributed targets, `interpolate!` may need the source grid
to also be distributed (same partition) or may handle CPU/single-GPU source
automatically. Test this -- if it fails, create the source grid with the same
`arch` as the target.

## Launch command

```bash
# Strip CUDA toolkit from LD_LIBRARY_PATH (see SIMULATION_NOTES.md #1)
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  mpiexecjl -n 8 julia -O0 --project=. simulations/ocean_24th_degree_8gpu.jl \
  simulations/checkpoints/ocean_tenth_degree_15day_jet35.jld2 \
  > simulations/logs/run_24th_$(date -u +%Y-%m-%dT%H-%M-%S).log 2>&1
```

Note: `mpiexecjl` is the Julia MPI launcher. May need `mpiexec` or `srun`
depending on the cluster scheduler. Ensure each rank gets one GPU
(CUDA_VISIBLE_DEVICES, --gpus-per-task=1, or equivalent).

## DO NOT use

- `using Reactant` or `using GordonBell25` -- poisons CUDA kernels
- `CPU()` source grids for interpolation onto `GPU()` targets
- Large dt without checking CFL against actual velocity magnitudes
- Dt > 12s at 1/24 degree with these velocity magnitudes

## Simulation plan

1. Load 1/10 degree checkpoint (u, v, T, S) from this run
2. Interpolate to 1/24 degree grid across 8 GPUs (Partition 4x2)
3. Run 15 days at Dt=12s (CFL ~0.2 at U_max=12 m/s)
4. Save u, v, T, S checkpoint
5. Total steps: 15 days / 12s = 108000 steps
6. Outer loops: 108000 / 64 = 1688 loops
7. Wall time estimate: depends on per-step throughput with 8-GPU communication.
   Single H200 at 1/10 does ~20s per 64-step loop. 1/24 has ~5.8x more points
   but 8 GPUs, so roughly ~20 * 5.8 / 8 ~ 14.5s per loop (optimistic).
   1688 loops * 14.5s ~ 6.8 hours (rough estimate).
