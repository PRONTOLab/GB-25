ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using GordonBell25
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

@inline exponential_profile(z, Lz, h) = (exp(z / h) - exp(-Lz / h)) / (1 - exp(-Lz / h))

function exponential_z_faces(; Nz, depth, h = Nz / 4.5)

    k = collect(1:Nz+1)
    z_faces = exponential_profile.(k, Nz, h)

    # Normalize
    z_faces .-= z_faces[1]
    z_faces .*= - depth / z_faces[end]

    z_faces[1] = 0.0

    return reverse(z_faces)
end

function simple_latitude_longitude_grid(arch, Nx, Ny, Nz; halo=(8, 8, 8))
    z = exponential_z_faces(; Nz, depth=4000) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo, z,
        longitude = (0, 360), # Problem is here: when longitude is not periodic we get error
        latitude = (-60, -30), # Tentative southern ocean latitude range
        topology = (Periodic, Bounded, Bounded)
    )

    return grid
end

function double_gyre_model(arch, Nx, Ny, Nz, Δt)

    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30)

    # TEOS10 is a 54-term polynomial that relates temperature (T) and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(equation_of_state = LinearEquationOfState())

    # Closures:
    vertical_closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()

    tracers = (:T, :S, :e)
    grid = simple_latitude_longitude_grid(arch, Nx, Ny, Nz)

    model = HydrostaticFreeSurfaceModel(; grid,
                                          free_surface = free_surface,
                                          closure = vertical_closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          momentum_advection = nothing,
                                          tracer_advection = nothing)

    set!(model.tracers.e, 1e-6)
    model.clock.last_Δt = Δt

    return model
end

function loop!(model)
    Δt = model.clock.last_Δt + 0
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace mincut = true track_numbers = false for i = 1:2
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

Nx = 362
Ny = 32
Nz = 30

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch  = ReactantState()
rmodel = double_gyre_model(rarch, Nx, Ny, Nz, 1200)

@info "Compiling..."
tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true loop!(rmodel)
compile_toc = time() - tic

@show compile_toc

@info "Running..."

tic = time()
restimate_tracer_error(rmodel)
@show rrun_toc
