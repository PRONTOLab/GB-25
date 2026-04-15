# Visualize distributed output using Oceananigans native tools.
# Reads per-rank JLD2 files and assembles onto a global grid.
#
# Usage: julia --project visualize_distributed.jl

using JLD2
using Printf
using Oceananigans
using Oceananigans.DistributedComputations: index2rank, rank2index

output_dir = joinpath("simulations", "output", "nccl_4gpu_3h")

# ── Reconstruct the global grid on CPU ────────────────────────────────

Nλ = 1440
Nφ = 560
Nz = 64
H  = 30e3

Oceananigans.defaults.FloatType = Float32

grid = LatitudeLongitudeGrid(CPU();
    size = (Nλ, Nφ, Nz),
    halo = (4, 4, 4),
    longitude = (0, 360),
    latitude = (-70, 70),
    z = (0, H))

@show grid

# ── Partition info ────────────────────────────────────────────────────

Rx, Ry, Rz = 2, 2, 1
local_Nx = Nλ ÷ Rx  # 720
local_Ny = Nφ ÷ Ry  # 280

# ── Load rank data and assemble onto global field ─────────────────────

function assemble_field(label, field_name)
    global_data = zeros(Float32, Nλ, Nφ, Nz)

    for r in 0:(Rx*Ry - 1)
        i, j, k = rank2index(r, Rx, Ry, Rz)  # 1-based (i, j, k)
        fname = joinpath(output_dir, @sprintf("state_rank%d_%s.jld2", r, label))

        local_data = JLD2.jldopen(fname, "r") do file
            file[field_name]
        end

        # Compute global offsets (1-based)
        x_offset = (i - 1) * local_Nx
        y_offset = (j - 1) * local_Ny

        # local_data from JLD2 is already interior (no halos)
        # Julia ordering: (Nλ_local, Nφ_local, Nz)
        global_data[x_offset+1:x_offset+local_Nx,
                    y_offset+1:y_offset+local_Ny,
                    1:Nz] .= local_data
    end

    return global_data
end

function assemble_to_field(label, field_name, loc)
    data = assemble_field(label, field_name)
    f = Field(loc, grid)
    copyto!(Oceananigans.interior(f), data)
    return f
end

# ── Assemble final state ──────────────────────────────────────────────

@info "Assembling final state..."
T_field  = assemble_to_field("final", "T",  (Center(), Center(), Center()))
u_field  = assemble_to_field("final", "u",  (Center(), Center(), Center()))
qv_field = assemble_to_field("final", "qᵛ", (Center(), Center(), Center()))

@info "Field extrema:" T=extrema(Oceananigans.interior(T_field)) u=extrema(Oceananigans.interior(u_field)) qv=extrema(Oceananigans.interior(qv_field))

# ── Save assembled global fields for plotting ─────────────────────────

output_file = joinpath(output_dir, "global_assembled.jld2")
JLD2.jldopen(output_file, "w") do file
    file["T"]  = Array(Oceananigans.interior(T_field))
    file["u"]  = Array(Oceananigans.interior(u_field))
    file["qv"] = Array(Oceananigans.interior(qv_field))
    file["lon"] = collect(grid.λᶜᵃᵃ[1:Nλ])
    file["lat"] = collect(grid.φᵃᶜᵃ[1:Nφ])
    file["z"]   = collect(grid.z.cᵃᵃᶜ[1:Nz])
end

@info "Saved assembled global fields to $output_file"
