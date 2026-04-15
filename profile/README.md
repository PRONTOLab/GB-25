# profile/

Post-processing of NVIDIA Nsight Compute (ncu) profiling reports for single-GPU roofline analysis on GH200.

## Workflow

1. **Run ncu profiling** via `sharding/alps_scaling_test.jl` (submits a SLURM job with `--set roofline --replay-mode application --profile-from-start off`). The `.ncu-rep` report is saved under `sharding/runs/<timestamp>_<hash>/ngpu=00001/`.

2. **Export CSV** — `ncu -i <report>.ncu-rep --csv --page raw` dumps all kernel metrics to a wide-format CSV.

3. **Analyze FLOPs** — `analyze_ncu_flops.jl` parses the CSV and prints a per-kernel breakdown (FP64/FP32/FP16 FLOPs, GFLOP/s, arithmetic intensity).

4. **Plot roofline** — `plot_roofline.jl` generates a log-log roofline PNG using CairoMakie, with GH200 peak ceilings (33.5 TFLOP/s FP64, 4 TB/s HBM3).

Steps 2–4 are automated by `run_ncu_analysis.sh`. Output (CSV + PNG) goes into a subfolder named after the run hash (e.g. `234_pJ4j/`).

## Quick start

```bash
cd profile/

# Full pipeline (requires uenv loaded for ncu binary):
bash run_ncu_analysis.sh

# Or run individual steps manually:
julia --project=. analyze_ncu_flops.jl 234_pJ4j/ncu_profile_raw_234_pJ4j.csv --top 20
julia --project=. plot_roofline.jl 234_pJ4j/ncu_profile_raw_234_pJ4j.csv --output 234_pJ4j/roofline_234_pJ4j.png --top 5
```

## Files

| File | Purpose |
|------|---------|
| `analyze_ncu_flops.jl` | Parse ncu CSV, compute FLOPs breakdown (supports both `--metrics` and `--set roofline` formats) |
| `plot_roofline.jl` | CairoMakie roofline plot (log-log scatter + BW/compute ceilings) |
| `run_ncu_analysis.sh` | End-to-end script: CSV export → analysis → plot |
| `Project.toml` | Julia project with CairoMakie dependency |

## Notes

- Multi-GPU ncu profiling is blocked by ncu 2025.2.1 lacking `--communicator shmem` (needs 2025.3+).
- The simulation uses `cuProfilerStart/Stop` to profile only the timestepping loop (not compilation).
