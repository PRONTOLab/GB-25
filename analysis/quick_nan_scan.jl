# Quick NaN scan across all checkpoints — only reads rank0, only T/ρ/u
using Serialization, Printf
root = ARGS[1]
ckpts = sort(filter(d -> startswith(d, "step_"),
                   readdir(root; join=false)))
for c in ckpts
    path = joinpath(root, c, "fields_rank0.dat")
    isfile(path) || continue
    state = open(Serialization.deserialize, path)
    s = ""
    for name in (:T, :ρ, :u, :w, :qᵛ, :qˡ)
        haskey(state, name) || continue
        info = state[name]
        arr = info.local_arrays[1]
        n = count(isnan, arr)
        f = filter(!isnan, arr)
        if isempty(f)
            s *= @sprintf("  %s:ALLNAN", String(name))
        else
            s *= @sprintf("  %s:[%+.2e, %+.2e]%s",
                          String(name), minimum(f), maximum(f),
                          n > 0 ? "(nan=$n)" : "")
        end
    end
    @printf("%-12s %s\n", c, s)
end
