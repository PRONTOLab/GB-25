ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics

using InteractiveUtils

using Reactant
using CUDA
#Reactant.set_default_backend("cpu")

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

using Enzyme


#
# Model parameters to set first:
#

# number of grid points
const Nx = 80  # LowRes: 48
const Ny = 160 # LowRes: 96
const Nz = 32

const halo_size = 4 #3 for non-immersed grid

#####
##### Model construction:
#####

function my_hydrostatic_velocity_fields()

    total_size = (Nx+2*halo_size) * (Ny+2*halo_size) * (Nz+2*halo_size)

    u = Float64.(reshape(1:total_size, Nx+2*halo_size, Ny+2*halo_size, Nz+2*halo_size))
    v = Float64.(reshape(1:total_size, Nx+2*halo_size, Ny+2*halo_size, Nz+2*halo_size))
    w = Float64.(reshape(1:total_size, Nx+2*halo_size, Ny+2*halo_size, Nz+2*halo_size))

    u = Reactant.to_rarray(u)
    v = Reactant.to_rarray(v)
    w = Reactant.to_rarray(w)

    return (u=u, v=v, w=w)
end

mutable struct my_Clock_struct{TT, DT, IT, S}
    time :: TT
    last_Δt :: DT
    last_stage_Δt :: DT
    iteration :: IT
    stage :: S
end

function Clock_helper(; time,
               last_Δt = Inf,
               last_stage_Δt = Inf,
               iteration = 0,
               stage = 1)

    TT = typeof(time)
    DT = typeof(last_Δt)
    IT = typeof(iteration)
    last_stage_Δt = convert(DT, last_Δt)
    return my_Clock_struct{TT, DT, IT, typeof(stage)}(time, last_Δt, last_stage_Δt, iteration, stage)
end

function my_Clock()
    FT = Float64
    t = ConcreteRNumber(zero(FT))
    iter = ConcreteRNumber(0)
    stage = 0
    last_Δt = convert(FT, Inf)
    last_stage_Δt = convert(FT, Inf)
    return Clock_helper(; time=t, iteration=iter, stage, last_Δt, last_stage_Δt)
end

function build_model()

    clock = my_Clock()

    velocities = my_hydrostatic_velocity_fields()
    
    model = MyModel(clock, velocities)

    return model
end

#####
##### Forward simulation (not actually using the Simulation struct)
#####


function estimate_tracer_error(model)
    model.clock.iteration = 0

    return nothing
end

function differentiate_tracer_error(model, dmodel)

    dedν = autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel))

    return dedν
end

mutable struct MyModel{C, U}
    clock::C
    velocities::U
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Make the grid:
model  = build_model()
dmodel = Enzyme.make_zero(model)

@info "Compiling the model run..."
tic = time()
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true  differentiate_tracer_error(model, dmodel)
compile_toc = time() - tic

@show compile_toc
