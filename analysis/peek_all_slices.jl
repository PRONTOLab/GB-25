using Serialization, Printf
dir = ARGS[1]
for rf in sort(filter(f->startswith(f,"fields_rank"), readdir(dir)))
    state = open(Serialization.deserialize, joinpath(dir, rf))
    println("=== $rf ===")
    info = state[:T]
    for (i, (arr, sl)) in enumerate(zip(info.local_arrays, info.local_slices))
        (ix, iy, iz) = sl
        nnan = count(isnan, arr)
        finite = filter(!isnan, arr)
        stat = isempty(finite) ? "ALL NaN" : @sprintf("fin min=%.3e max=%.3e mean=%.3e nan=%d/%d", minimum(finite), maximum(finite), sum(Float64.(finite))/length(finite), nnan, length(arr))
        @printf("  T shard[%d] x=%s y=%s z=%s size=%s  %s\n", i, string(ix), string(iy), string(iz), string(size(arr)), stat)
    end
end
