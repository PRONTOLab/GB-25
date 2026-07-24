# profile/

Post-processing of profiling data for roofline analysis on GH200. Two complementary approaches:

- **ncu** (Nsight Compute): Hardware-counter roofline from a single-GPU run — true FP64/FP32 FLOP counts, DRAM bytes, per-kernel arithmetic intensity.
- **XLA** (Reactant.Profiler): Model-based roofline from `xplane.pb` traces — program-level and per-op metrics across multi-GPU scaling runs (4→1024 GPUs).

## ncu Workflow

1. **Run ncu profiling** via `sharding/alps_scaling_test.jl` (submits a SLURM job with `--set roofline --replay-mode application --profile-from-start off`). The `.ncu-rep` report is saved under `sharding/runs/<timestamp>_<hash>/ngpu=00001/`.

2. **Export CSV** — `ncu -i <report>.ncu-rep --csv --page raw` dumps all kernel metrics to a wide-format CSV.

3. **Analyze FLOPs** — `analyze_ncu_flops.jl` parses the CSV and prints a per-kernel breakdown (FP64/FP32/FP16 FLOPs, GFLOP/s, arithmetic intensity).

4. **Plot roofline** — `plot_roofline.jl` generates a log-log roofline PNG using CairoMakie, with GH200 peak ceilings (33.5 TFLOP/s FP64, 4 TB/s HBM3).

Steps 2–4 are automated by `run_ncu_analysis.sh`. Output (CSV + PNG) goes into a subfolder named after the run hash (e.g. `234_pJ4j/`).

## XLA Roofline Workflow

`analyze_xla_roofline.jl` extracts roofline data from `xplane.pb` files (produced by `Reactant.Profiler.with_profiler`) using `Reactant.Profiler.get_total_program_roofline()` and `get_framework_op_stats()`.

It outputs:
- **Program-level roofline table** — XLA FLOP rate, memory BW, BW utilization, operational intensity, roofline efficiency, and bound classification per GPU count.
- **Scaling summary** — ncu-extrapolated GFLOP/ts and aggregate PFLOP/s across GPU counts.
- **Per-op breakdown** — Top framework ops by self-time with per-op FLOP rate, BW, intensity, and bound.
- **Bound classification** — Time-weighted HBM vs Compute breakdown.

```bash
cd profile/

# Analyze a scaling run (auto-discovers all ngpu= subdirectories):
julia --project=.. analyze_xla_roofline.jl ../sharding/runs/2026-04-09T00-19-49.653_5CCn

# Specific GPU counts and phase:
julia --project=.. analyze_xla_roofline.jl ../sharding/runs/<run_dir> --ngpus 4 1024 --phase loop2

# Custom ncu reference:
julia --project=.. analyze_xla_roofline.jl ../sharding/runs/<run_dir> --ncu-gflop 154.691 --ncu-grid 752x752x64 --top-ops 20
```

### ncu vs XLA comparison (single GPU / 4-GPU node)

| Metric | ncu (1 GPU) | XLA (4 GPUs, per-GPU) |
|--------|---:|---:|
| Total FLOPs/ts | 154.7 GFLOP (HW FP64+FP32) | ~21,674 GFLOP (HLO model) |
| Arithmetic Intensity | 0.48 FLOP/byte | 79.6 FLOP/byte |
| Memory BW | 2,377 GB/s | 1,846 GB/s |
| BW Utilization | 63% | 49% |
| Wall time / ts | 136.7 ms | 137.4 ms |

XLA model FLOPs are ~140× larger than ncu FP64 FLOPs because XLA counts every HLO operation (selects, broadcasts, indexing, comparisons), not just floating-point instructions. Wall time and memory BW are directly comparable. Both agree the workload is memory-bound.

## Quick start

```bash
cd profile/

# ncu pipeline (requires uenv loaded for ncu binary):
bash run_ncu_analysis.sh

# Or run individual steps manually:
julia --project=. analyze_ncu_flops.jl 234_pJ4j/ncu_profile_raw_234_pJ4j.csv --top 20
julia --project=. plot_roofline.jl 234_pJ4j/ncu_profile_raw_234_pJ4j.csv --output 234_pJ4j/roofline_234_pJ4j.png --top 5

# XLA roofline (requires Reactant from parent project):
julia --project=.. analyze_xla_roofline.jl ../sharding/runs/2026-04-09T00-19-49.653_5CCn
```

## Files

| File | Purpose |
|------|---------|
| `analyze_ncu_flops.jl` | Parse ncu CSV, compute FLOPs breakdown (supports both `--metrics` and `--set roofline` formats) |
| `plot_roofline.jl` | CairoMakie roofline plot (log-log scatter + BW/compute ceilings) |
| `run_ncu_analysis.sh` | End-to-end script: CSV export → analysis → plot |
| `analyze_xla_roofline.jl` | Extract XLA roofline from `xplane.pb` via Reactant.Profiler (multi-GPU scaling) |
| `analyze_xla_traces.py` | Parse `trace.json.gz` for compute/NCCL timing breakdown (Python) |
| `Project.toml` | Julia project with CairoMakie dependency |

## Notes

- Multi-GPU ncu profiling is blocked by ncu 2025.2.1 lacking `--communicator shmem` (needs 2025.3+).
- The simulation uses `cuProfilerStart/Stop` to profile only the timestepping loop (not compilation).
