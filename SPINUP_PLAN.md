# 1-Year Spin-Up Plan: Tripolar Global Ocean

## Goal

Run a 1-year spin-up of a global ocean simulation on a **TripolarGrid** with
realistic Earth bathymetry, ECCO2 initial conditions, and JRA55 atmospheric
forcing. The end state will serve as a physically-equilibrated initial
condition for subsequent science and benchmarking runs.

## Why TripolarGrid?

A `LatitudeLongitudeGrid` is singular at the poles (Δx → 0) which forces
extremely small time steps and creates numerical issues in the Arctic.
A `TripolarGrid` displaces the northern singularities to Greenland and
Siberia where they sit on land (and are therefore masked by bathymetry),
which permits much larger time steps and lets us actually represent the
Arctic Ocean.

This is the production grid for global ocean climate simulations.

## Configuration

### Grid

- **Type:** `TripolarGrid` (from `OrthogonalSphericalShellGrids`)
- **Horizontal resolution:** 1/4° (target). Start at 1/2° for shakedown, then
  scale up.
  - 1/2°: Nx=720, Ny=340
  - 1/4°: Nx=1440, Ny=680
- **Vertical:** Nz=40, `exponential_z_faces(; Nz, depth=6000, h=30)` —
  6 km depth with surface refinement
- **Halo:** (7, 7, 7)
- **Bathymetry:** ETOPO via `regrid_bathymetry(underlying_grid)` →
  `ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))`

### Physics

Uses ClimaOcean/NumericalEarth defaults from `ocean_simulation`:

- Hydrostatic primitive equations, free surface
- **Free surface:** `SplitExplicitFreeSurface` with auto-computed substeps
- **Time stepper:** `SplitRungeKutta3`
- **Momentum advection:** `WENOVectorInvariant` (5th order)
- **Tracer advection:** `WENO(order=7)`
- **Closure:** `CATKEVerticalDiffusivity` (default ocean closure)
- **Equation of state:** `TEOS10EquationOfState`
- **Coriolis:** `HydrostaticSphericalCoriolis`
- **Bottom drag:** quadratic, Cd = 0.003

### Forcing

- **Atmosphere:** `JRA55PrescribedAtmosphere` (real reanalysis,
  3-hourly winds/temperature/humidity/radiation/precipitation)
- **Air-sea fluxes:** `SimilarityTheoryFluxes` with `FixedIterations(5)`
- **Radiation:** `Radiation(arch)` — longwave + shortwave with realistic
  ocean albedo

### Initial conditions

- **T, S:** `ECCOMetadata(:temperature/:salinity; dates=DateTime(1993,1,1))`
  via `set!(ocean.model, T=..., S=...)` — initialized from ECCO4 monthly
  reanalysis on Jan 1, 1993
- **Velocities:** zero (let the model spin up its own currents)
- **Free surface η:** zero

### Polar restoring

Restore T and S in the polar regions (where dynamics are unconstrained by
the prescribed atmosphere) toward the ECCO climatology:

- **Mask:** `LinearlyTaperedPolarMask(southern=(-80,-70), northern=(70,90))`
- **Rate:** 1/(7 days)
- **Source:** `ECCOMetadata(:temperature/:salinity, dates, ECCO4Monthly())`
  with monthly dates spanning 1993

### Time stepping

- **Δt at 1/2°:** 10 minutes (target ~5–10 SYPD on 4 GPUs)
- **Δt at 1/4°:** 5 minutes
- **Stop time:** 365 days (1 year)

## Architecture

- **4 GPUs (A100-80GB)** in this Studio
- **Distributed:** `Distributed(GPU(); partition = Partition(2, 2, 1))` for 4 GPUs
- **NCCL** for halo communication (required for GPU-aware MPI)
- **Communication:** Use `OceananigansNCCLExt` via `using NCCL`

We do **not** use Reactant for the spin-up — we want straightforward,
restartable, observable behavior. Reactant runs come later for the
benchmarking phase.

## Outputs

Save sufficient state to (a) restart, (b) make movies, and (c) diagnose
the spin-up state.

### Surface fields (every 1 day)

```
outputs = merge(ocean.model.velocities, ocean.model.tracers)
JLD2OutputWriter(ocean.model, outputs;
                 filename = "spinup_surface.jld2",
                 indices = (:, :, Nz),
                 schedule = TimeInterval(1day),
                 overwrite_existing = false)  # so we can resume
```

### Full 3D state checkpoints (every 30 days)

```
Checkpointer(ocean.model;
             prefix = "spinup_checkpoint",
             schedule = TimeInterval(30days),
             overwrite_existing = false,
             cleanup = false)  # keep all checkpoints for restart safety
```

### Zonal-mean diagnostics (every 1 day)

For monitoring spin-up — global mean SST, zonal-mean T/S, AMOC index,
ACC transport. Implement as a `Callback` writing to a small JLD2.

### Progress callback (every 100 iterations)

Print: simulation time, iteration, max|u|, T extrema, wall time, SYPD.

## Restart strategy

The simulation should be **restartable from checkpoint**. Two reasons:

1. A 1-year run at ~5 SYPD takes ~73 wall hours — too long for one job.
2. Diagnostics (e.g., new output writers, callback frequency) can be added
   between segments.

Wrap the simulation logic so that:

- If a checkpoint file exists, load from it and continue
- Otherwise, build the model from scratch and initialize from ECCO

## Spin-up monitoring

Track these signals to know when the model is "spun up enough":

1. **Global KE drift** — should level off after ~6 months
2. **SST drift vs ECCO climatology** — should be small (<2°C globally averaged)
3. **AMOC overturning streamfunction** — should reach ~15 Sv at 26°N
4. **ACC transport at Drake Passage** — should reach ~130–150 Sv
5. **Max |u|** — should be physical (<3 m/s away from coastlines)

## Execution plan

### Phase 1: Shakedown (1/2°, ~1–2 days wall time)

1. Build the model on 4 GPUs at 1/2°
2. Run for 30 days, verify stability and reasonable physics
3. Inspect output surface fields and diagnostics
4. Tune Δt if needed (max stable + safe CFL)

**Deliverable:** Surface SST/speed plot at day 30, looking like a
recognizable global ocean.

### Phase 2: 1/2° year-long spin-up

1. Continue from Phase 1 checkpoint
2. Run for 365 days total
3. Save daily surface fields, monthly checkpoints, daily zonal means
4. Generate monthly snapshot plots throughout the run

**Deliverable:** Spun-up 1/2° state at day 365, plus monthly
visualization of the spin-up.

### Phase 3 (optional): 1/4° refinement

Use the Phase 2 end state, interpolated onto a 1/4° grid, as the
initial condition for a shorter (60–90 day) 1/4° run. This gives us
realistic eddying initial conditions for the high-resolution
benchmarks.

## Files to create

1. `simulations/ocean_climate_spinup.jl` — main run script
   - Builds tripolar grid + bathymetry on distributed GPU arch
   - Loads ECCO ICs and sets up restoring
   - Configures JRA55 atmosphere
   - Sets up checkpointer and output writers
   - Detects existing checkpoint and resumes if present
   - Runs the simulation
2. `simulations/visualize_spinup.jl` — post-processing
   - Reads surface JLD2 + checkpoint files
   - Generates SST/speed/MLD plots
   - Computes zonal-mean drift, AMOC, ACC transport
3. `SPINUP_PLAN.md` — this document
4. (Once Phase 1 succeeds) `simulations/ocean_climate_spinup_quarter.jl`
   for the optional 1/4° refinement

## Open questions / decisions before starting

1. **Δt**: Start with 10 min at 1/2°? Or be more conservative at 5 min?
   The default `estimate_maximum_Δt` for 1/2° gives ~30 min — but with
   Earth bathymetry and CATKE we usually want to be well under that.

2. **Substeps for SplitExplicit**: Use the grid-aware constructor
   `SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=Δt)` so it
   computes the right number of substeps automatically.

3. **Polar restoring**: Use the `LinearlyTaperedPolarMask` from
   ClimaOcean/NumericalEarth or roll our own?

4. **Checkpoint format**: JLD2 (Oceananigans default) is fine, but
   we should verify checkpoint→restart works end-to-end before
   committing to a long run.

5. **Sea ice**: For the 1-year spin-up do we want sea ice? `OceanSeaIceModel`
   with `ClimaSeaIce` is the way to go. Skipping it (use `OceanOnlyModel`)
   means high latitudes will get unrealistic surface heat exchange but
   lets us avoid sea ice complexity. Recommend: skip for Phase 1
   shakedown, add for Phase 2 if it's already wired up in NumericalEarth.

## Notes

- The cached files `bathymetry_sixth_degree.jld2` and
  `ecco2_initial_conditions_sixth_degree.jld2` from earlier work
  are **not** what we want here — they're 1/6° on a LatLon grid. The
  spin-up uses ETOPO bathymetry directly via `regrid_bathymetry` on the
  TripolarGrid, and ECCOMetadata for the ICs.
- The 4-GPU partition `(2, 2, 1)` matches the NCCL distributed setup
  we tested earlier. NCCL extension must be loaded via `using NCCL`.
- Watch for the same NCCL corner-buffer issue we hit before with
  `ImmersedBoundaryGrid` on a `Distributed{GPU}` arch — our patched
  Oceananigans branch handles it, but verify before launching the
  long run.
