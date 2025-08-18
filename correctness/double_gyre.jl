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

Nx = 362
Ny = 32
Nz = 30

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch  = ReactantState()
rmodel = double_gyre_model(rarch, Nx, Ny, Nz, 1200)

using InteractiveUtils

using KernelAbstractions: @kernel, @index

using Oceananigans.Grids: static_column_depthᶜᶜᵃ

using Oceananigans.Operators: ℑzᵃᵃᶠ

using Oceananigans.TimeSteppers: update_state!

using Oceananigans.TurbulenceClosures: compute_diffusivities!, getclosure

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: compute_CATKE_diffusivities!, mask_diffusivity, κuᶜᶜᶠ, κcᶜᶜᶠ, κeᶜᶜᶠ, momentum_mixing_lengthᶜᶜᶠ, convective_length_scaleᶜᶜᶠ, turbulent_velocityᶜᶜᶜ, stability_functionᶜᶜᶠ, stable_length_scaleᶜᶜᶠ

using Oceananigans.Utils: launch!

callbacks   = []
closure     = rmodel.closure
diffusivity = rmodel.diffusivity_fields

@show @which compute_diffusivities!(diffusivity, closure, rmodel; parameters = :xyz)

function bad_compute_diffusivities!(diffusivities, closure, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock

    launch!(arch, grid, parameters,
            bad_compute_CATKE_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function bad_compute_CATKE_diffusivities!(diffusivities, grid, closure, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    Jᵇ = diffusivities.Jᵇ

    # Note: we also compute the TKE diffusivity here for diagnostic purposes, even though it
    # is recomputed in time_step_turbulent_kinetic_energy.
    κu★ = bad_κuᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ)
    κu★ = mask_diffusivity(i, j, k, grid, κu★)

    @inbounds diffusivities.κu[i, j, k] = κu★
end

@inline function bad_κuᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κu = bad_momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    κu_max = closure.maximum_viscosity
    κu★ = min(κu, κu_max)
    FT = eltype(grid)
    return FT(κu★)
end

@inline function bad_momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, surface_buoyancy_flux)
    Cᶜ  = closure.mixing_length.Cᶜu
    Cᵉ  = closure.mixing_length.Cᵉu
    Cˢᵖ = closure.mixing_length.Cˢᵖ
    ℓʰ = convective_length_scaleᶜᶜᶠ(i, j, k, grid, closure, Cᶜ, Cᵉ, Cˢᵖ, velocities, tracers, buoyancy, surface_buoyancy_flux)

    Cᵘⁿ = closure.mixing_length.Cᵘⁿu
    Cˡᵒ = closure.mixing_length.Cˡᵒu
    Cʰⁱ = closure.mixing_length.Cʰⁱu
    σ = stability_functionᶜᶜᶠ(i, j, k, grid, closure, Cᵘⁿ, Cˡᵒ, Cʰⁱ, velocities, tracers, buoyancy)

    ℓ★ = σ * stable_length_scaleᶜᶜᶠ(i, j, k, grid, closure, tracers.e, velocities, tracers, buoyancy)

    ℓʰ = ifelse(isnan(ℓʰ), zero(grid), ℓʰ)
    ℓ★ = ifelse(isnan(ℓ★), zero(grid), ℓ★)
    ℓu = max(ℓ★, ℓʰ)

    H = static_column_depthᶜᶜᵃ(i, j, grid)
    return min(H, ℓu)
end

@info "Compiling..."
tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true bad_compute_diffusivities!(diffusivity, closure, rmodel; parameters = :xyz)
compile_toc = time() - tic

@show compile_toc

@info "Running..."

tic = time()
restimate_tracer_error(diffusivity, closure, rmodel)
@show rrun_toc
