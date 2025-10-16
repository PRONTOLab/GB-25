
using Random
using Reactant


# number of grid points
const Nx = 48 #96  # LowRes: 48
const Ny = 96 #192 # LowRes: 96
const Nz = 32

#####
##### Forward simulation (not actually using the Simulation struct)
#####

function loop!(random_field)

    @trace mincut = true track_numbers = false for i = 1:16
        # Get some new randomness:
        randn!(random_field)
    end
    return nothing
end

random_field  = Reactant.to_rarray(zeros(Nx, Ny, 1)) #Field{Center, Center, Nothing}(model.grid)

@info "Compiling the model run..."
tic = time()
rloop! = @compile raise_first=true raise=true sync=true loop!(random_field)
compile_toc = time() - tic

@show compile_toc

@info "Running the simulation..."

rloop!(random_field)