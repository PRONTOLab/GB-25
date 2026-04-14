# Diagnose NaN extent in extracted slices
using Printf
Nx, Ny = 2880, 1280
viz = ARGS[1]
for f in readdir(viz; join=true)
    endswith(f, ".f32") || continue
    bytes = read(f)
    a = reinterpret(Float32, bytes)
    @assert length(a) == Nx*Ny
    nnan = count(isnan, a)
    na   = length(a) - nnan
    finite_vals = filter(!isnan, a)
    if isempty(finite_vals)
        @printf("%-24s  ALL NaN (%d cells)\n", basename(f), length(a))
    else
        mn = minimum(finite_vals); mx = maximum(finite_vals); me = sum(Float64.(finite_vals))/length(finite_vals)
        @printf("%-24s  nan=%d/%d (%.2f%%)  finite: min=% .3e max=% .3e mean=% .3e\n",
                basename(f), nnan, length(a), 100*nnan/length(a), mn, mx, me)
    end
end
