# /home/avik-pal/.julia/bin/mpiexecjl -np 4 --project=. julia --threads=32 --color=yes --startup=no GB-25/sharding/simple_sharding_problem.jl

ENV["CUDA_VISIBLE_DEVICES"] = ""
# ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using MPI
MPI.Init()  # Only needed if using MPI to detect the coordinator

using Oceananigans
using Reactant

Reactant.Distributed.initialize()

ndevices = length(Reactant.devices())
nxdevices = floor(Int, sqrt(ndevices))
nydevices = ndevices ÷ nxdevices

arch = Oceananigans.Distributed(
    Oceananigans.ReactantState();
    partition=Partition(nxdevices, nydevices, 1)
)
Nx = Ny = 16
Nz = 4
grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=(0, 1))

# display(grid)

# exit(0)

# function mtn₁(λ, φ)
#     λ₁ = 70
#     φ₁ = 55
#     dφ = 5
#     return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
# end

# function mtn₂(λ, φ)
#     λ₁ = 70
#     λ₂ = λ₁ + 180
#     φ₂ = 55
#     dφ = 5
#     return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
# end

# gaussian_islands(λ, φ) = 2 * (mtn₁(λ, φ) + mtn₂(λ, φ))
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_islands)) # xxx?

model = HydrostaticFreeSurfaceModel(; grid)
model.clock.last_Δt = 60
compiled_first_time_step! = @compile optimize=true raise=true Oceananigans.TimeSteppers.first_time_step!(model)
compiled_first_time_step!(model)

# c = CenterField(grid);
# ∇²c = Field(∂x(∂x(c)) + ∂y(∂y(c)));

# cᵢ(λ, φ, z) = exp(-(λ^2 + φ^2) / 200)
# set!(c, cᵢ)

# Oceananigans.BoundaryConditions.fill_halo_regions!(c)

# res = @jit compute!(∇²c);

# if Reactant.Distributed.local_rank() == 0
hlo = @code_hlo compute!(∇²c);
# end

if Reactant.Distributed.local_rank() == 0
    println(hlo)
end

# parent(∇²c)


# #### Run on CPU

# cpu_arch = CPU()
# grid_cpu = TripolarGrid(cpu_arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=(0, 1))

# c_cpu = CenterField(grid_cpu);
# ∇²c_cpu = Field(∂x(∂x(c_cpu)) + ∂y(∂y(c_cpu)));

# res_cpu = compute!(∇²c_cpu);


# #### Run with ReactantState
# using Reactant, Oceananigans

# Nx = Ny = Nz = 128

# r_arch = Oceananigans.ReactantState()
# grid_r = TripolarGrid(r_arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=(0, 1))

# c_r = CenterField(grid_r);
# ∇²c_r = Field(∂x(∂x(c_r)) + ∂y(∂y(c_r)));

# res_r = @jit compute!(∇²c_r);


# # Resharding Problem
# using Reactant

# struct MyStruct{A}
#     a::A
# end

# mesh = Sharding.Mesh(reshape(Reactant.devices(), 2, :), (:x, :y))

# x = Reactant.to_rarray(MyStruct(rand(2, 2)))
# y = Reactant.to_rarray(
#     rand(2, 2); sharding=Sharding.NamedSharding(mesh, (:x, :y))
# )

# function fn(a::MyStruct, b)
#     a.a .= a.a .+ b
#     return a
# end
