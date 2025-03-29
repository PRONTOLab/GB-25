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

using Dates
@info "[$rank] Generating model..." now(UTC)

grid_str = get(ENV, "grid_name", "simple_lat_lon")
resolution_fraction_str = get(ENV, "resolution_fraction", "2")
time_step_str = get(ENV, "time_step", "60")
Nz_str = get(ENV, "Nz", "10")

@show grid_type = Symbol(grid_str)
@show resolution_fraction = parse(Float64, resolution_fraction_str)
@show time_step_str = parse(Float64, time_step_str)
@show Nz = parse(Int, Nz_str)

model = GordonBell25.baroclinic_instability_model(arch; grid_type, Δt=1, Nz,
                                                  resolution=1/resolution_fraction)

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

