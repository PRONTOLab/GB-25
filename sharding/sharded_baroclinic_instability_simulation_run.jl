using GordonBell25
using GordonBell25: first_time_step!, time_step!, loop!
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using Reactant
using MPI

# Need this for sharding with non-openMPI implementations?
# (GHA uses MPICH)
MPI.Init()

Reactant.Distributed.initialize(; single_gpu_per_process=false)

@show Ndev = length(Reactant.devices())

if Ndev == 1
    rank = 0
    arch = Oceananigans.ReactantState()
elseif Ndev == 2
    rank = Reactant.Distributed.local_rank()

    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(1, 2, 1)
    )
else
    Rx = floor(Int, sqrt(Ndev))
    Ry = Ndev ÷ Rx
    rank = Reactant.Distributed.local_rank()

    arch = Oceananigans.Distributed(
        Oceananigans.ReactantState();
        partition = Partition(Rx, Ry, 1)
    )
end

using Dates
@info "[$rank] Generating model..." now(UTC)

H = 8
Rx, Ry = GordonBell25.factors(Ndev)
Tx = 16 * Rx
Ty = 16 * Ry
Nz = 16

Nx = Tx - 2H
Ny = Ty - 2H

model = GordonBell25.baroclinic_instability_model(arch, Nx, Ny, Nz; halo=(H, H, H), Δt=1)

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

