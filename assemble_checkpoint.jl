# Assemble per-rank 1/16° output into a single global JLD2 checkpoint.

using JLD2

output_dir = joinpath(@__DIR__, "simulations", "output", "nccl_8gpu_16th_deg")
iter_str = "036000"
Rx, Ry = 4, 2

# Reference sizes from rank 0
ref_path = joinpath(output_dir, "fields_rank0_iter$(iter_str).jld2")
ref_sizes = JLD2.jldopen(ref_path, "r") do f
    Dict(k => size(f[k]) for k in keys(f) if f[k] isa AbstractArray)
end

# Center-field ref size
ref_nx, ref_ny, ref_nz = ref_sizes["ρ"]
Nλ = Rx * ref_nx
Nφ = Ry * ref_ny
Nz = ref_nz

@info "Assembling" Nλ Nφ Nz ref_nx ref_ny ref_nz

function assemble(name)
    @info "  Assembling $name..."
    tiles = Vector{Array{Float32, 3}}(undef, Rx * Ry)
    for r in 0:(Rx * Ry - 1)
        path = joinpath(output_dir, "fields_rank$(r)_iter$(iter_str).jld2")
        tiles[r + 1] = JLD2.jldopen(path, "r") do f; f[name]; end
    end

    # Trim face fields to center-field tile size
    nx, ny, nz = ref_nx, ref_ny, ref_nz
    g = zeros(Float32, Rx * nx, Ry * ny, nz)
    for r in 0:(Rx * Ry - 1)
        ix = r ÷ Ry; iy = r % Ry
        t = tiles[r + 1]
        g[ix*nx+1:(ix+1)*nx, iy*ny+1:(iy+1)*ny, :] = t[1:nx, 1:ny, 1:nz]
    end
    return g
end

outpath = joinpath(@__DIR__, "simulations", "initial_conditions", "sixteenth_deg_12h_assembled.jld2")

JLD2.jldopen(outpath, "w") do f
    f["Nλ"] = Nλ
    f["Nφ"] = Nφ
    f["Nz"] = Nz
    f["ρ"]  = assemble("ρ")
    f["ρu"] = assemble("ρu")
    f["ρv"] = assemble("ρv")
    f["ρw"] = assemble("ρw")
    f["ρθ"] = assemble("ρθ")
    f["ρqᵛ"] = assemble("ρqᵛ")
    f["micro_ρqᶜˡ"] = assemble("ρqᶜˡ")
    f["micro_ρqᶜⁱ"] = assemble("ρqᶜⁱ")
    f["ρqʳ"] = assemble("ρqʳ")
    f["ρqˢ"] = assemble("ρqˢ")
end

@info "Saved" outpath filesize(outpath) / 1e9
