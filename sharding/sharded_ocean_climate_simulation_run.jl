using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using Reactant
using MPI

using Dates
@info "This is when the fun begins" now(UTC)

# Need this for sharding with non-openMPI implementations?
# (GHA uses MPICH)
MPI.Init()

if !(get(ENV, "CI", "false") == "true")
    Reactant.Distributed.initialize(; single_gpu_per_process=false)
end

@show Ngpu = length(Reactant.devices())

if Ngpu == 1
    rank = 0
    arch = Oceananigans.ReactantState()
elseif Ngpu == 2
    rank = Reactant.Distributed.local_rank()

    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(1, 2, 1)
    )
else
    Rx = floor(Int, sqrt(Ngpu))
    Ry = Ngpu ÷ Rx
    rank = Reactant.Distributed.local_rank()

    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(Rx, Ry, 1)
    )
end

@info "[$rank] Generating model..." now(UTC)
@info "[$rank] allocations" GordonBell25.allocatorstats()
H = 8
Tx = 48 * Rx
Ty = 24 * Ry
Nz = 4

Nx = Tx - 2H
Ny = Ty - 2H

model = GordonBell25.data_free_ocean_climate_model_init(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=10)

@info "[$rank] Compiling first_time_step!..." 
rfirst! = @compile first_time_step!(model)

@info "[$rank] Compiling loop..."
rstep! = @compile time_step!(model)

@time "[$rank] Running first_time_step!..." rfirst!(model)
@time "[$rank] Warming up..." rstep!(model)

rstep!(model)
rstep!(model)
rstep!(model)

@time "[$rank] Running loop..." begin
    for n = 1:10
        rstep!(model)
    end
end
