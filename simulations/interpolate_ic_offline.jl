#=
Offline interpolation of the cached baroclinic instability initial condition
onto a smaller grid whose **total** field size (interior + halos) is
768 × 768 × 64. With halo = (1, 1, 1), interior = (766, 766, 62), so
parent(T) has shape (768, 768, 64).

Why: at runtime, `set_baroclinic_instability_from_file!` skips the slow
CPU-side intermediate interpolation when the source's total field shape is
an integer factor of the target's total field shape. Pre-baking the IC to
a 768³-ish total grid lets any target with parent dims that are multiples
of (768, 768, 64) hit the fast path.

Output:
    simulations/initial_conditions/baroclinic_100day_768cube.jld2

Run:
    julia --project -O0 simulations/interpolate_ic_offline.jl
=#

using JLD2
using Oceananigans
using Oceananigans.Fields: CenterField, interpolate!
using Oceananigans.BoundaryConditions: fill_halo_regions!

const SRC = joinpath(@__DIR__, "initial_conditions", "baroclinic_100day_quarter_degree.jld2")
const DST = joinpath(@__DIR__, "initial_conditions", "baroclinic_100day_768cube.jld2")

# Target *total* (parent) shape:
const HALO       = (1, 1, 1)
const TOTAL      = (768, 768, 64)
const INTERIOR   = TOTAL .- 2 .* HALO   # (766, 766, 62)

@info "Loading source IC" SRC
Nx_src, Ny_src, Nz_src, T_data, S_data = JLD2.jldopen(SRC, "r") do f
    (f["Nx"], f["Ny"], f["Nz"], f["T"], f["S"])
end
@info "Source dims" Nx_src Ny_src Nz_src size_T=size(T_data)

src_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
    size = (Nx_src, Ny_src, Nz_src),
    halo = (1, 1, 1),
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)
T_src = CenterField(src_grid)
S_src = CenterField(src_grid)
set!(T_src, T_data)
set!(S_src, S_data)
fill_halo_regions!(T_src)
fill_halo_regions!(S_src)

@info "Building destination grid" INTERIOR HALO TOTAL
dst_grid = LatitudeLongitudeGrid(Oceananigans.CPU();
    size = INTERIOR,
    halo = HALO,
    z = (-4000, 0),
    latitude = (-80, 80),
    longitude = (0, 360),
)
T_dst = CenterField(dst_grid)
S_dst = CenterField(dst_grid)

@info "Interpolating T..."
interpolate!(T_dst, T_src)
@info "Interpolating S..."
interpolate!(S_dst, S_src)
fill_halo_regions!(T_dst)
fill_halo_regions!(S_dst)

T_total = Array(parent(T_dst))
S_total = Array(parent(S_dst))
@assert size(T_total) == TOTAL "T parent shape $(size(T_total)) ≠ $TOTAL"
@assert size(S_total) == TOTAL "S parent shape $(size(S_total)) ≠ $TOTAL"

@info "Writing destination IC" DST extrema_T=extrema(T_total) extrema_S=extrema(S_total)
JLD2.jldopen(DST, "w") do f
    f["Nx"]    = INTERIOR[1]
    f["Ny"]    = INTERIOR[2]
    f["Nz"]    = INTERIOR[3]
    f["halo"]  = HALO
    f["total"] = TOTAL
    # Stored as PARENT (total) arrays — interior + halos
    f["T_total"] = T_total
    f["S_total"] = S_total
end
@info "Done" filesize=filesize(DST)
