using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using SeawaterPolynomials
using Reactant
using Random

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
resolution_fraction_str = get(ENV, "resolution_fraction", "4")
@show resolution_fraction = parse(Float64, resolution_fraction_str)
grid = :gaussian_islands # an idealized TripolarGrid with Gaussian-shaped islands
model = GordonBell25.baroclinic_instability_model(arch; resolution=1/resolution_fraction, Δt=1, grid)

@info "[$rank] Compiling first_time_step!..." 
rfirst! = @compile first_time_step!(model)

@info "[$rank] Compiling loop..."
rstep! = @compile time_step!(model)

@time "[$rank] Running first_time_step!..." rfirst!(model)
@time "[$rank] Warming up..." rstep!(model)

rstep!(model)
rstep!(model)
rstep!(model)

@time "[$rank] Running loop" begin
    for n = 1:10
        @time "[$rank] Time step $n" rstep!(model)
    end
end
