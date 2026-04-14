# Extract 2D slices from a sharded checkpoint and write raw Float32 binaries
# usage: julia --startup-file=no extract_surface_slices.jl <checkpoint_dir> <output_dir>
#
# For each field we emit a 2880x1280 (interior) surface slice + a mid-level (k=32) slice.
# Clouds get a column-integrated cloud-water-path proxy (sum over all z of qˡ+qᶜˡ+qⁱ+qᶜⁱ).

using Serialization
using Printf

const Hx, Hy, Hz = 4, 4, 4
const Nx_full, Ny_full, Nz_full = 2888, 1288, 72
const Nx,       Ny,       Nz       = Nx_full - 2Hx, Ny_full - 2Hy, Nz_full - 2Hz  # 2880, 1280, 64

ckpt_dir = ARGS[1]
out_dir  = ARGS[2]
mkpath(out_dir)

# Which fields to slice. For cloud path we'll build from individual water fields later.
surface_fields = [:T, :u, :v, :w, :qᵛ, :ρ, :θ]
midlev_fields  = [:T, :u, :v, :w, :qᵛ]
path_fields    = [:qˡ, :qᶜˡ, :qⁱ, :qᶜⁱ]   # for cloud water path
rain_fields    = [:qʳ, :qˢ]                # precip path
k_mid = 32 + Hz   # interior k=32, includes halo offset

# Storage for assembly (halo-inclusive global)
function global_xy(T::Type=Float32)
    zeros(T, Nx_full, Ny_full)
end

# surf (k = 1 + Hz), mid (k = k_mid) slices per field
surf = Dict{Symbol, Matrix{Float32}}()
mid  = Dict{Symbol, Matrix{Float32}}()
for f in surface_fields; surf[f] = global_xy(); end
for f in midlev_fields;  mid[f]  = global_xy(); end

# column-integrated paths (2D)
cloud_path = global_xy()
rain_path  = global_xy()

rank_files = sort(filter(f -> startswith(f, "fields_rank") && endswith(f, ".dat"),
                         readdir(ckpt_dir)))
@info "Found $(length(rank_files)) rank files"

k_surf = 1 + Hz

for rf in rank_files
    path = joinpath(ckpt_dir, rf)
    t0 = time()
    @info "Deserializing $rf..."
    state = open(Serialization.deserialize, path)
    dt_read = time() - t0
    @info @sprintf("  read+deserialize: %.1fs", dt_read)

    # Per shard, copy the z-slices into the global arrays at their slice bounds
    for name in union(surface_fields, midlev_fields, path_fields, rain_fields)
        haskey(state, name) || continue
        info = state[name]
        for (arr, sl) in zip(info.local_arrays, info.local_slices)
            (ixlo, ixhi), (iylo, iyhi), (izlo, izhi) = sl
            # Surface slice at k = k_surf (global halo-inclusive index)
            if k_surf >= izlo && k_surf <= izhi && (name in surface_fields)
                kloc = k_surf - izlo + 1
                @views surf[name][ixlo:ixhi, iylo:iyhi] .= arr[:, :, kloc]
            end
            # Mid-level slice at k = k_mid
            if k_mid >= izlo && k_mid <= izhi && (name in midlev_fields)
                kloc = k_mid - izlo + 1
                @views mid[name][ixlo:ixhi, iylo:iyhi] .= arr[:, :, kloc]
            end
            # Column sum over INTERIOR z cells only (1+Hz : Nz_full-Hz)
            if (name in path_fields) || (name in rain_fields)
                z_int_lo = max(1+Hz, izlo)
                z_int_hi = min(Nz_full-Hz, izhi)
                if z_int_lo <= z_int_hi
                    zlo_loc = z_int_lo - izlo + 1
                    zhi_loc = z_int_hi - izlo + 1
                    # sum across z
                    sum_slab = dropdims(sum(arr[:, :, zlo_loc:zhi_loc], dims=3), dims=3)
                    if name in path_fields
                        @views cloud_path[ixlo:ixhi, iylo:iyhi] .+= sum_slab
                    else
                        @views rain_path[ixlo:ixhi, iylo:iyhi]  .+= sum_slab
                    end
                end
            end
        end
    end
    state = nothing
    GC.gc()
end

# Strip halos → interior 2880×1280
function strip_halo(A)
    A[Hx+1:Hx+Nx, Hy+1:Hy+Ny]
end

function dump(name, A)
    Aint = strip_halo(A)
    # column-major Float32 raw; Python can np.fromfile(...).reshape((Ny, Nx), order='F')
    open(joinpath(out_dir, "$(name).f32"), "w") do io
        write(io, Float32.(Aint))
    end
    @printf("  wrote %-16s  shape=%s  min=% .4e  max=% .4e  mean=% .4e\n",
            name*".f32", string(size(Aint)),
            Float64(minimum(Aint)), Float64(maximum(Aint)), Float64(sum(Aint)/length(Aint)))
end

@info "Writing outputs to $out_dir"
for (f, A) in surf; dump("surf_"*String(f), A); end
for (f, A) in mid;  dump("mid_"*String(f),  A); end
dump("cloud_path",  cloud_path)
dump("rain_path",   rain_path)

# Manifest
open(joinpath(out_dir, "manifest.txt"), "w") do io
    println(io, "Nx=", Nx)
    println(io, "Ny=", Ny)
    println(io, "dtype=float32 order=fortran  # Julia column-major")
    println(io, "checkpoint=", ckpt_dir)
    println(io, "halo=(", Hx, ",", Hy, ",", Hz, ")")
    println(io, "k_surf (interior)=1")
    println(io, "k_mid (interior)=32")
end

@info "Done."
