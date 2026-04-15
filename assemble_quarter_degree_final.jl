# Assemble per-rank prognostic fields from the 4-GPU quarter-degree final state
# into a single global IC file for initializing the eighth-degree run.
#
# Usage: julia --project assemble_quarter_degree_final.jl

using JLD2
using Printf

output_dir = joinpath("simulations", "output", "nccl_4gpu_3h")
ic_output  = joinpath("simulations", "initial_conditions", "quarter_deg_3h_final.jld2")

# ── Partition and grid info (must match the 4-GPU run) ────────────────
Rx, Ry, Rz = 2, 2, 1
Nλ, Nφ, Nz = 1440, 560, 64
local_Nx = Nλ ÷ Rx  # 720
local_Ny = Nφ ÷ Ry  # 280

# Oceananigans rank ordering: r = (i-1)*Ry*Rz + (j-1)*Rz + (k-1)
function rank2index(r)
    i = div(r, Ry * Rz) + 1
    r_rem = mod(r, Ry * Rz)
    j = div(r_rem, Rz) + 1
    k = mod(r_rem, Rz) + 1
    return (i, j, k)
end

# Assemble center-located field: each rank contributes (local_Nx, local_Ny, Nz)
function assemble_center(field_name)
    global_data = zeros(Float32, Nλ, Nφ, Nz)
    for r in 0:(Rx*Ry - 1)
        i, j, _ = rank2index(r)
        fname = joinpath(output_dir, @sprintf("state_rank%d_final.jld2", r))
        local_data = JLD2.jldopen(fname, "r") do f
            f[field_name]
        end
        @assert size(local_data) == (local_Nx, local_Ny, Nz) "Unexpected size for $field_name on rank $r: $(size(local_data))"
        x_off = (i - 1) * local_Nx
        y_off = (j - 1) * local_Ny
        global_data[x_off+1:x_off+local_Nx, y_off+1:y_off+local_Ny, :] .= local_data
    end
    @info "Assembled $field_name" size=size(global_data) extrema=extrema(global_data)
    return global_data
end

# Assemble YFace field: j=1 ranks have (local_Nx, local_Ny, Nz),
#                        j=Ry ranks have (local_Nx, local_Ny+1, Nz)
function assemble_yface(field_name)
    global_data = zeros(Float32, Nλ, Nφ + 1, Nz)
    for r in 0:(Rx*Ry - 1)
        i, j, _ = rank2index(r)
        fname = joinpath(output_dir, @sprintf("state_rank%d_final.jld2", r))
        local_data = JLD2.jldopen(fname, "r") do f
            f[field_name]
        end
        local_ny = size(local_data, 2)
        x_off = (i - 1) * local_Nx
        y_off = (j - 1) * local_Ny  # face offset aligned with cell offset
        global_data[x_off+1:x_off+local_Nx, y_off+1:y_off+local_ny, :] .= local_data
        @info "  rank $r (i=$i, j=$j): $field_name local_ny=$local_ny → y[$( y_off+1):$(y_off+local_ny)]"
    end
    @info "Assembled $field_name (YFace)" size=size(global_data) extrema=extrema(global_data)
    return global_data
end

# Assemble ZFace field: all ranks contribute (local_Nx, local_Ny, Nz+1)
function assemble_zface(field_name)
    global_data = zeros(Float32, Nλ, Nφ, Nz + 1)
    for r in 0:(Rx*Ry - 1)
        i, j, _ = rank2index(r)
        fname = joinpath(output_dir, @sprintf("state_rank%d_final.jld2", r))
        local_data = JLD2.jldopen(fname, "r") do f
            f[field_name]
        end
        @assert size(local_data) == (local_Nx, local_Ny, Nz + 1) "Unexpected size for $field_name on rank $r: $(size(local_data))"
        x_off = (i - 1) * local_Nx
        y_off = (j - 1) * local_Ny
        global_data[x_off+1:x_off+local_Nx, y_off+1:y_off+local_Ny, :] .= local_data
    end
    @info "Assembled $field_name (ZFace)" size=size(global_data) extrema=extrema(global_data)
    return global_data
end

# ── Assemble all prognostic fields ────────────────────────────────────
@info "Assembling quarter-degree final state from 4-GPU output..."

ρ    = assemble_center("ρ")
ρu   = assemble_center("ρu")     # XFace with periodic lon → same size as center
ρv   = assemble_yface("ρv")      # YFace on bounded lat
ρw   = assemble_zface("ρw")      # ZFace on bounded z
ρθ   = assemble_center("ρθ")
ρqv  = assemble_center("ρqᵛ")
ρqcl = assemble_center("ρqᶜˡ")
ρqci = assemble_center("ρqᶜⁱ")

# Read time/iteration from rank 0
time_val, iter_val = JLD2.jldopen(joinpath(output_dir, "state_rank0_final.jld2"), "r") do f
    (f["time"], f["iteration"])
end
@info "Source state" time=time_val iteration=iter_val

# ── Save ──────────────────────────────────────────────────────────────
@info "Saving assembled IC to $ic_output..."
JLD2.jldsave(ic_output;
    Nλ  = Nλ,
    Nφ  = Nφ,
    Nz  = Nz,
    ρ   = ρ,
    ρu  = ρu,
    ρv  = ρv,
    ρw  = ρw,
    ρθ  = ρθ,
    ρqᵛ = ρqv,
    micro_ρqᶜˡ = ρqcl,
    micro_ρqᶜⁱ = ρqci,
    time      = time_val,
    iteration = iter_val)

@info "Done" path=ic_output filesize=filesize(ic_output)
