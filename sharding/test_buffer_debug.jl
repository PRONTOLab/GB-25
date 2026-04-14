using Dates
@info "Buffer debugging test" now(UTC)

using GordonBell25
using GordonBell25: first_time_step!, loop!, factors, is_distributed_env_present
using Oceananigans
using Oceananigans.TimeSteppers: update_state!

const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 64, grid_y_default = 64, grid_z_default = 64)

Oceananigans.defaults.FloatType = GordonBell25.float_type_from_args(parsed_args)

using Oceananigans.Architectures: ReactantState
using Reactant

if !is_distributed_env_present()
    using MPI
    MPI.Init()
end

GordonBell25.preamble()
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.WHILE_CONCAT[] = true
GordonBell25.initialize(; single_gpu_per_process=false)

local_arch = ReactantState()
arch = local_arch
Ndev = length(Reactant.devices())
Rx, Ry = factors(Ndev)
if Ndev == 1; rank = 0
else
    arch = Oceananigans.Distributed(arch; partition = Partition(Rx, Ry, 1))
    rank = Reactant.Distributed.local_rank()
end

H = 4
Nλ = parsed_args["grid-x"] * Rx - 2H
Nφ = parsed_args["grid-y"] * Ry - 2H
Nz = parsed_args["grid-z"]

model = GordonBell25.moist_baroclinic_wave_model(arch;
    Nλ, Nφ, Nz, H=30e3, Δt=0.8, halo=(H, H, 4))

function probe(rank, label, model)
    @info "[$rank] === $label ==="
    fields = Oceananigans.fields(model)
    for name in (:ρ, :ρu, :ρθ)
        haskey(fields, name) || continue
        f = fields[name]
        ifrt = Reactant.ancestor(f)
        oid = objectid(ifrt.data)
        la, _, _ = GordonBell25.local_shards_to_host(ifrt)
        @info "[$rank] $name: oid=$oid extrema=$(extrema(la[1]))"
    end
end

probe(rank, "BEFORE COMPILE", model)

co = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=:all)
@info "[$rank] Compiling first_time_step!..." now(UTC)
rfirst! = @time @compile compile_options=co first_time_step!(model)

probe(rank, "AFTER COMPILE", model)

@info "[$rank] Executing first_time_step!..." now(UTC)
@time "[$rank] exec" rfirst!(model)

probe(rank, "AFTER EXECUTION", model)

# IDEA 1: Reactant.synchronize
@info "[$rank] === IDEA 1: synchronize ==="
for name in (:ρ, :ρu)
    f = Oceananigans.fields(model)[name]
    ifrt = Reactant.ancestor(f)
    try
        Reactant.synchronize(ifrt)
        la, _, _ = GordonBell25.local_shards_to_host(ifrt)
        @info "[$rank] SYNC $name: extrema=$(extrema(la[1]))"
    catch e
        @warn "[$rank] sync failed: $name" e
    end
end

# IDEA 2: interior()
@info "[$rank] === IDEA 2: interior ==="
for name in (:ρ, :ρu)
    f = Oceananigans.fields(model)[name]
    try
        idata = Oceananigans.interior(f)
        ifrt = Reactant.ancestor(idata)
        la, _, _ = GordonBell25.local_shards_to_host(ifrt)
        @info "[$rank] interior($name): extrema=$(extrema(la[1]))"
    catch e
        @warn "[$rank] interior failed: $name" e
    end
end

# IDEA 3: Array(interior(f)) — the report_state path that worked at 2 nodes
@info "[$rank] === IDEA 3: Array(interior(f)) ==="
for name in (:ρ, :ρu, :ρθ)
    f = Oceananigans.fields(model)[name]
    try
        data = Array(Oceananigans.interior(f))
        @info "[$rank] Array(interior($name)): size=$(size(data)) extrema=$(extrema(data))"
    catch e
        @warn "[$rank] Array(interior) failed: $name" e
    end
end

# IDEA 4: compile+run update_state! (dkz pattern) then probe
@info "[$rank] === IDEA 4: update_state! ==="
rupdate! = @time @compile compile_options=co update_state!(model, compute_tendencies=false)
@time "[$rank] update_state!" rupdate!(model)
probe(rank, "AFTER update_state!", model)

@info "[$rank] Done!" now(UTC)
