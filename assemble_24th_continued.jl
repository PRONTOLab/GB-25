# Assemble a per-rank checkpoint from the 1/24° continuation run into one JLD2.
# Pass ITER as env var (e.g. ITER=4500).

using JLD2

iter_num = parse(Int, get(ENV, "ITER", "4500"))
iter_str = lpad(iter_num, 6, "0")
output_dir = joinpath(@__DIR__, "simulations", "output", "nccl_8gpu_24th_deg_continued")
Rx, Ry = 4, 2

ref_path = joinpath(output_dir, "fields_rank0_iter$(iter_str).jld2")
isfile(ref_path) || error("Missing ref: $ref_path")
ref_nx, ref_ny, ref_nz = JLD2.jldopen(ref_path, "r") do f; size(f["ρ"]); end
@info "Per-rank center shape" ref_nx ref_ny ref_nz iter_str

Nλ = Rx * ref_nx; Nφ = Ry * ref_ny; Nz = ref_nz
@info "Global" Nλ Nφ Nz

function assemble(name)
    @info "  $name…"
    tiles = Vector{Array{Float32, 3}}(undef, Rx * Ry)
    for r in 0:(Rx * Ry - 1)
        path = joinpath(output_dir, "fields_rank$(r)_iter$(iter_str).jld2")
        tiles[r + 1] = JLD2.jldopen(path, "r") do f; f[name]; end
    end
    nx = ref_nx
    tile0_ny = size(tiles[1], 2)
    has_extra_y = any(size(tiles[r+1], 2) > tile0_ny for r in 0:(Rx*Ry-1))
    global_ny = has_extra_y ? Ry * ref_ny + 1 : Ry * ref_ny
    global_nz = size(tiles[1], 3)
    g = zeros(Float32, Rx * nx, global_ny, global_nz)
    for r in 0:(Rx * Ry - 1)
        ix = r ÷ Ry; iy = r % Ry
        t = tiles[r + 1]
        tny = size(t, 2); tnz = size(t, 3)
        y_start = iy * ref_ny + 1
        g[ix*nx+1:(ix+1)*nx, y_start:y_start+tny-1, 1:tnz] = t[1:nx, :, :]
    end
    @info "    → $(size(g))"
    return g
end

outpath = joinpath(@__DIR__, "simulations", "initial_conditions",
                   "twentyfourth_continued_iter$(iter_num)_assembled.jld2")
@info "Writing" outpath

JLD2.jldopen(outpath, "w") do f
    f["Nλ"] = Nλ; f["Nφ"] = Nφ; f["Nz"] = Nz
    f["iteration"] = JLD2.jldopen(ref_path, "r") do g; g["iteration"]; end
    f["time"]      = JLD2.jldopen(ref_path, "r") do g; g["time"]; end
    f["Δt"]        = JLD2.jldopen(ref_path, "r") do g; g["Δt"]; end
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

@info "Saved" outpath round(filesize(outpath) / 1e9, digits=1)
