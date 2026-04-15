# Extract 6 2D slices from the latest 1/24° continued checkpoint into a small
# native JLD2 file (Julia-friendly serialization).
#
# Run: ITER=9000 julia --project=. extract_viz_slices.jl

using JLD2

iter_num = parse(Int, get(ENV, "ITER", "9000"))
src = joinpath(@__DIR__, "simulations", "initial_conditions",
               "twentyfourth_continued_iter$(iter_num)_assembled.jld2")
dst = joinpath(@__DIR__, "viz_slices_iter$(iter_num).jld2")
isfile(src) || error("Missing source: $src")

k_mid = 15           # ~7 km (Δz = 30000/64 = 468.75 m → z(k=15) ≈ 7.27 km)
Δz    = 30000.0 / 64.0

@info "Reading" src k_mid

ρ_sfc, ρu_sfc, ρv_sfc_full, ρqv_sfc, ρ_mid, ρqcl_mid, ρw_mid, ρqr_volume =
    JLD2.jldopen(src, "r") do f
        Nx, Ny, Nz = size(f["ρ"])
        @info "Grid" Nx Ny Nz
        (f["ρ"][:, :, 1],                   # surface ρ
         f["ρu"][:, :, 1],                  # Periodic x-Face → same (Nx, Ny)
         f["ρv"][:, :, 1],                  # Face-y Bounded → (Nx, Ny+1)
         f["ρqᵛ"][:, :, 1],                 # (Nx, Ny)
         f["ρ"][:, :, k_mid + 1],           # mid-level ρ (Julia 1-based, so index k_mid+1)
         f["micro_ρqᶜˡ"][:, :, k_mid + 1],  # mid-level cloud liquid
         f["ρw"][:, :, k_mid + 1],          # Face-z, mid-level
         f["ρqʳ"])                          # full 3D for column integration
    end

Nx, Ny = size(ρ_sfc)
ρv_sfc = ρv_sfc_full[:, 1:Ny]  # drop the wall face row

safe_sfc = max.(ρ_sfc, 0.001f0)
safe_mid = max.(ρ_mid, 0.001f0)

u_sfc   = Float32.(ρu_sfc  ./ safe_sfc)
v_sfc   = Float32.(ρv_sfc  ./ safe_sfc)
qv_sfc  = Float32.(ρqv_sfc ./ safe_sfc .* 1000f0)     # g/kg
qcl_mid = Float32.(ρqcl_mid ./ safe_mid .* 1000f0)    # g/kg
w_mid   = Float32.(ρw_mid  ./ safe_mid)               # m/s

# Column-integrated rain: Σ_k ρqʳ[:,:,k] * Δz → kg/m²
@info "Integrating ρqʳ vertically…"
qr_vertint = Float32.(dropdims(sum(ρqr_volume; dims=3); dims=3) .* Δz)  # (Nx, Ny)

lat = Float32.(range(-80, 80; length=Ny + 1)[1:Ny] .+ (160.0f0 / Ny / 2))  # cell centers
lon = Float32.(range(0, 360; length=Nx + 1)[1:Nx] .+ (360.0f0 / Nx / 2))

@info "Writing" dst
JLD2.jldopen(dst, "w") do f
    f["iteration"]   = iter_num
    f["sim_time_h"]  = iter_num * 0.8 / 3600
    f["k_mid"]       = k_mid
    f["z_mid_km"]    = (k_mid + 0.5) * Δz / 1000
    f["dz_m"]        = Δz
    f["source"]      = basename(src)
    f["lon_deg"]     = lon
    f["lat_deg"]     = lat
    f["u_sfc"]       = u_sfc
    f["v_sfc"]       = v_sfc
    f["qv_sfc"]      = qv_sfc
    f["qr_vertint"]  = qr_vertint
    f["qcl_mid"]     = qcl_mid
    f["w_mid"]       = w_mid
end

println("  u_sfc   range: [", extrema(u_sfc), "]")
println("  v_sfc   range: [", extrema(v_sfc), "]")
println("  qv_sfc  range: [", extrema(qv_sfc), "]  (g/kg)")
println("  qr_vertint range: [", extrema(qr_vertint), "]  (kg/m²)")
println("  qcl_mid range: [", extrema(qcl_mid), "]  (g/kg)")
println("  w_mid   range: [", extrema(w_mid), "]  (m/s)")

sz = filesize(dst) / 1e6
@info "Saved" dst round(sz; digits=1)
