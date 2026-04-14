using Serialization, Printf
const Hx, Hy, Hz = 4, 4, 4
ckpt = ARGS[1]
state = open(Serialization.deserialize, joinpath(ckpt, "fields_rank0.dat"))

for name in (:T, :ρ, :u, :v, :w, :θ, :qᵛ, :qˡ, :ρu, :ρθ)
    haskey(state, name) || continue
    info = state[name]
    arr = info.local_arrays[1]
    Nx, Ny, Nz = size(arr)
    interior = arr[Hx+1:Nx-Hx, Hy+1:Ny-Hy, Hz+1:Nz-Hz]
    @printf("%-5s  INTERIOR %dx%dx%d: min=% .4e max=% .4e mean=% .4e  nan=%d zeros=%d\n",
            String(name), size(interior)..., minimum(interior), maximum(interior),
            sum(Float64.(interior))/length(interior),
            count(isnan, interior), count(iszero, interior))
end

# Also check interior z=1 (surface) specifically for rho
for name in (:ρ, :T, :θ)
    haskey(state, name) || continue
    arr = state[name].local_arrays[1]
    Nx, Ny, Nz = size(arr)
    surf = arr[Hx+1:Nx-Hx, Hy+1:Ny-Hy, Hz+1]  # first interior z
    topp = arr[Hx+1:Nx-Hx, Hy+1:Ny-Hy, Nz-Hz] # last interior z
    @printf("%-5s  surface z=1:  min=% .4e max=% .4e mean=% .4e  zeros=%d\n",
            String(name), minimum(surf), maximum(surf), sum(Float64.(surf))/length(surf), count(iszero, surf))
    @printf("%-5s  top   z=Nz:   min=% .4e max=% .4e mean=% .4e  zeros=%d\n",
            String(name), minimum(topp), maximum(topp), sum(Float64.(topp))/length(topp), count(iszero, topp))
end
