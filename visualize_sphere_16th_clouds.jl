# Visualize 1/16° on sphere: w, cloud ice, surface rain.

using JLD2
using Oceananigans
using CairoMakie

output_dir = joinpath(@__DIR__, "simulations", "output", "nccl_8gpu_16th_deg")
iter_str = "036000"

Rx, Ry = 4, 2
Nλ_global = 5760
Nφ_global = 2560
Nz = 64
H = 30e3

function load_and_assemble(name, iter_str, output_dir, Rx, Ry)
    tiles = Vector{Array{Float32, 3}}(undef, Rx * Ry)
    for r in 0:(Rx * Ry - 1)
        path = joinpath(output_dir, "fields_rank$(r)_iter$(iter_str).jld2")
        tiles[r + 1] = JLD2.jldopen(path, "r") do f; f[name]; end
    end
    # Use rank 0 center-field size as reference; trim face fields
    ref = size(tiles[1])
    nx, ny, nz = ref
    g = zeros(Float32, Rx * nx, Ry * ny, nz)
    for r in 0:(Rx * Ry - 1)
        ix = r ÷ Ry; iy = r % Ry
        t = tiles[r + 1]
        g[ix*nx+1:(ix+1)*nx, iy*ny+1:(iy+1)*ny, :] = t[1:nx, 1:ny, 1:nz]
    end
    return g
end

@info "Loading fields..."
ρ_data = load_and_assemble("ρ", iter_str, output_dir, Rx, Ry)
ρw_data = load_and_assemble("ρw", iter_str, output_dir, Rx, Ry)
ρqci_data = load_and_assemble("ρqᶜⁱ", iter_str, output_dir, Rx, Ry)
ρqr_data = load_and_assemble("ρqʳ", iter_str, output_dir, Rx, Ry)

FT = Float32
safe_ρ = max.(ρ_data, FT(0.001))
w_data = ρw_data[:, :, 1:Nz] ./ safe_ρ
qci_data = ρqci_data ./ safe_ρ .* FT(1000)
qr_data = max.(ρqr_data, FT(0)) ./ safe_ρ .* FT(1000)

grid = LatitudeLongitudeGrid(CPU();
    size = (Nλ_global, Nφ_global, Nz),
    halo = (4, 4, 4),
    latitude = (-80, 80),
    longitude = (0, 360),
    z = (0, H))

w_field = CenterField(grid)
qci_field = CenterField(grid)
qr_field = CenterField(grid)

@info "Setting field data..."
Oceananigans.interior(w_field) .= FT.(w_data)
Oceananigans.interior(qci_field) .= FT.(qci_data)
Oceananigans.interior(qr_field) .= FT.(qr_data)

@info "Rendering..."

k_mid = Nz ÷ 2
k_sfc = 1
elev = 0.4; azim = 1.2

fig = Figure(size = (1800, 700))

# w mid-level
ax1 = Axis3(fig[1, 1]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax1); hidespines!(ax1)
plt1 = surface!(ax1, view(w_field, :, :, k_mid); colormap = :balance, colorrange = (-1, 1))
Colorbar(fig[2, 1], plt1; vertical = false, width = Relative(0.6), label = "w [m/s]")

# Cloud ice mid-level
ax2 = Axis3(fig[1, 2]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax2); hidespines!(ax2)
plt2 = surface!(ax2, view(qci_field, :, :, k_mid); colormap = cgrad([:black, :white]), colorrange = (0, 30))
Colorbar(fig[2, 2], plt2; vertical = false, width = Relative(0.6), label = "qᶜⁱ [g/kg]")

# Surface rain
ax3 = Axis3(fig[1, 3]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax3); hidespines!(ax3)
plt3 = surface!(ax3, view(qr_field, :, :, k_sfc); colormap = cgrad([:midnightblue, :white]), colorrange = (0, 1))
Colorbar(fig[2, 3], plt3; vertical = false, width = Relative(0.6), label = "qʳ surface [g/kg]")

Label(fig[0, :], "1/16° (iter $iter_str, sim 12h) — w, cloud ice, surface rain", fontsize = 20)

colgap!(fig.layout, -100)
rowgap!(fig.layout, 1, -50)
rowsize!(fig.layout, 1, Relative(0.8))

outpath = joinpath(@__DIR__, "sphere_sixteenth_degree_clouds.png")
save(outpath, fig; px_per_unit = 4)
@info "Saved" outpath
