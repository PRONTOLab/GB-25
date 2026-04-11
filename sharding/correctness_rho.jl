#!/usr/bin/env julia
#
# Compare InterpolateArray (Reactant) vs plain CPU nearest-neighbor on ρ only.
#
# Run:
#   XLA_FLAGS="--xla_force_host_platform_device_count=4" julia --project=. sharding/correctness_rho.jl

ENV["XLA_FLAGS"] = get(ENV, "XLA_FLAGS", "") * " --xla_force_host_platform_device_count=4"

using JLD2
using Reactant
using Reactant: Sharding, InterpolateArray, InterpolationType

Ndev = length(Reactant.devices())
@info "Devices" Ndev
@assert Ndev >= 4

mesh = Sharding.Mesh(reshape(Reactant.devices()[1:4], 2, 2), (:x, :y))
sharding_3d = Sharding.NamedSharding(mesh, ("x", "y", nothing))

ic_path = joinpath(@__DIR__, "..", "simulations", "initial_conditions",
                   "atmosphere_coarsened_32x16x8.jld2")
ρ_src = Float32.(JLD2.load(ic_path, "ρ"))

halo = (4, 4, 4)
Nx, Ny, Nz = 120, 120, 8
target_size = (Nx + 2halo[1], Ny + 2halo[2], Nz + 2halo[3])

@info "ρ source" size=size(ρ_src) range=extrema(ρ_src)
@info "Target" target_size halo interior=(Nx, Ny, Nz)

# Reactant path
r_full = Array(InterpolateArray(ρ_src, target_size, sharding_3d, InterpolationType.Nearest, halo))
hx, hy, hz = halo
r_int = r_full[hx+1:end-hx, hy+1:end-hy, hz+1:end-hz]

# CPU nearest-neighbor (matches _nearest_neighbor_data_copy! kernel)
Nx_s, Ny_s, Nz_s = size(ρ_src)
Rx = Nx / Nx_s
Ry = Ny / Ny_s
v_int = zeros(Float32, Nx, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    i′ = ceil(Int, i / Rx)
    j′ = ceil(Int, j / Ry)
    v_int[i, j, k] = ρ_src[i′, j′, k]
end

@info "Reactant interior" size=size(r_int) range=extrema(r_int)
@info "CPU interior"      size=size(v_int) range=extrema(v_int)
@info "Max abs diff" maximum(abs.(r_int .- v_int))
@info "Match?" r_int≈v_int

println("\n— Reactant interior[:, 60, 4] —")
println(r_int[:, 60, 4])
println("\n— CPU interior[:, 60, 4] —")
println(v_int[:, 60, 4])
