using Serialization
using Printf

dir = ARGS[1]
rank0 = joinpath(dir, "fields_rank0.dat")
@info "Reading $rank0"
state = open(Serialization.deserialize, rank0)

@printf("iteration = %s\n", repr(get(state, :iteration, nothing)))
@printf("time      = %s\n", repr(get(state, :time, nothing)))
@printf("halo_sizes= %s\n", repr(get(state, :halo_sizes, nothing)))
@printf("z_indices = %s\n", repr(get(state, :z_indices, nothing)))
@printf("slices    = %s\n", repr(get(state, :slices, nothing)))

fn = state[:field_names]
@printf("field_names (%d):\n", length(fn))
for name in fn
    info = state[name]
    la = info.local_arrays
    gs = info.global_shape
    shape0 = size(la[1])
    slices0 = info.local_slices[1]
    @printf("  %-8s  global=%s  nlocal=%d  shape[1]=%s  slice[1]=%s  eltype=%s\n",
            String(name), string(gs), length(la), string(shape0),
            string(slices0), string(eltype(la[1])))
end
