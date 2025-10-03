```sh
ENABLE_JITPROFILING=1 perf record -F 99 -o abernathy.data --call-graph dwarf -k 1 ~/src/julia-1.11/julia --project=. -g2 correctness/abernathey_channel.jl
perf inject --jit --input abernathy.data --output abernathy-jit.data
```

```sh
perf script -i abernathy-short-jit.data | stackcollapse-perf.pl --inline --all > out-short.perf-folded
flamegraph.pl out.perf-folded > perf.svg
```

### Short profile
```
perf record -p PID -o abernathy-short.data --call-graph dwarf -k 1
perf inject --jit --input abernathy-short.data --output abernathy-short-jit.data
```

## Firefox (https://profiler.firefox.com)
```
perf script -F +pid --input abernathy-short-4-jit.data > abernathy.perf
```

## Inferno
perf script | inferno-collapse-perf > stacks.folded
cat stacks.folded | inferno-flamegraph > flamegraph.