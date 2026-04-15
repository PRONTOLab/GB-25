# Visualize 1/16° distributed output on sphere: surface θ, qv, wind speed.
# Assembles per-rank JLD2 files into global fields.

using JLD2
using Oceananigans
using CairoMakie

# ── Config ────────────────────────────────────────────────────────────

output_dir = joinpath(@__DIR__, "simulations", "output", "nccl_8gpu_16th_deg")
iter_str = "036000"  # final checkpoint (sim 12h)

Rx, Ry = 4, 2
Nλ_global = 5760
Nφ_global = 2560
Nz = 64

# ── Load and assemble per-rank data ──────────────────────────────────

function load_and_assemble(name, iter_str, output_dir, Rx, Ry)
    tiles = Vector{Array{Float32, 3}}(undef, Rx * Ry)
    for r in 0:(Rx * Ry - 1)
        path = joinpath(output_dir, "fields_rank$(r)_iter$(iter_str).jld2")
        tiles[r + 1] = JLD2.jldopen(path, "r") do f
            f[name]
        end
    end

    nx, ny, nz = size(tiles[1])
    global_field = zeros(Float32, Rx * nx, Ry * ny, nz)

    for r in 0:(Rx * Ry - 1)
        ix = r ÷ Ry
        iy = r % Ry
        global_field[ix*nx+1:(ix+1)*nx, iy*ny+1:(iy+1)*ny, :] = tiles[r + 1]
    end

    return global_field
end

@info "Loading ρ..."
ρ_data = load_and_assemble("ρ", iter_str, output_dir, Rx, Ry)
@info "Loading ρθ..."
ρθ_data = load_and_assemble("ρθ", iter_str, output_dir, Rx, Ry)
@info "Loading ρqᵛ..."
ρqv_data = load_and_assemble("ρqᵛ", iter_str, output_dir, Rx, Ry)
@info "Loading ρu..."
ρu_data = load_and_assemble("ρu", iter_str, output_dir, Rx, Ry)
@info "Global shape" size(ρ_data)

# ── Derive fields ─────────────────────────────────────────────────────
# Skip ρv assembly (face field has Nφ+1) — compute wind speed per-rank
# using only ρu for approximate |u| (dominant component in jets).

FT = Float32
safe_ρ = max.(ρ_data, FT(0.001))
θ_data = ρθ_data ./ safe_ρ
qv_data = ρqv_data ./ safe_ρ .* FT(1000)
u_data = ρu_data ./ safe_ρ
wspd_data = abs.(u_data)  # approximate wind speed from zonal component

# ── Build CPU grid + fields ───────────────────────────────────────────

grid = LatitudeLongitudeGrid(CPU();
    size = (Nλ_global, Nφ_global, Nz),
    halo = (4, 4, 4),
    latitude = (-80, 80),
    longitude = (0, 360),
    z = (0, 30e3))

θ_field = CenterField(grid)
qv_field = CenterField(grid)
wspd_field = CenterField(grid)

@info "Setting field data..."
Oceananigans.interior(θ_field) .= FT.(θ_data)
Oceananigans.interior(qv_field) .= FT.(qv_data)
Oceananigans.interior(wspd_field) .= FT.(wspd_data)

# ── Render ────────────────────────────────────────────────────────────

@info "Rendering..."

k_sfc = 1
elev = 0.4
azim = 1.2

fig = Figure(size = (1800, 700))

ax1 = Axis3(fig[1, 1]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax1); hidespines!(ax1)
plt1 = surface!(ax1, view(θ_field, :, :, k_sfc); colormap = Reverse(:RdYlBu), colorrange = (250, 315))
Colorbar(fig[2, 1], plt1; vertical = false, width = Relative(0.6), label = "θ [K]")

ax2 = Axis3(fig[1, 2]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax2); hidespines!(ax2)
plt2 = surface!(ax2, view(qv_field, :, :, k_sfc); colormap = :tempo, colorrange = (0, 34))
Colorbar(fig[2, 2], plt2; vertical = false, width = Relative(0.6), label = "qv [g/kg]")

ax3 = Axis3(fig[1, 3]; aspect = :data, elevation = elev, azimuth = azim, protrusions = 0)
hidedecorations!(ax3); hidespines!(ax3)
plt3 = surface!(ax3, view(wspd_field, :, :, k_sfc); colormap = :solar, colorrange = (0, 30))
Colorbar(fig[2, 3], plt3; vertical = false, width = Relative(0.6), label = "Wind speed [m/s]")

Label(fig[0, :], "1/16° (iter $iter_str, sim 12h) — surface fields on the sphere", fontsize = 20)

colgap!(fig.layout, -100)
rowgap!(fig.layout, 1, -50)
rowsize!(fig.layout, 1, Relative(0.8))

outpath = joinpath(@__DIR__, "sphere_sixteenth_degree.png")
save(outpath, fig; px_per_unit = 4)
@info "Saved" outpath
