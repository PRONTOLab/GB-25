using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using Reactant

using ClimaOcean
using ClimaOcean.OceanSeaIceModels.InterfaceComputations: FixedIterations, ComponentInterfaces
using OrthogonalSphericalShellGrids: TripolarGrid

using CFTime
using Dates
using Printf
using Profile

PROFILE::Base.RefValue{Bool} = Ref(false)

macro gbprofile(name::String, expr::Expr)
    return quote
        if $(PROFILE)[]
            $(Profile.clear)()
            $(Profile.init)(; delay=0.1)
            out = $(Profile).@profile $(esc(expr))
            open(string("profile_", $(esc(name)), ".txt"), "w") do s
                println(s, "# Showing profile of")
                println(s, "#     ", $(string(expr)))
                println(s, "# at ", $(string(__source__)))
                $(Profile.print)(IOContext(s, :displaysize => (48, 1000)))
            end
            $(Profile.clear)()
            out
        else
            $(esc(expr))
        end
    end
end

# Utility for printing progress to the terminal
wall_time::UInt64 = UInt64(0)

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    Tmax = maximum(interior(T))
    Tmin = minimum(interior(T))
    umax = (maximum(abs, interior(u)), maximum(abs, interior(v)), maximum(abs, interior(w)))
    step_time = 1e-9 * (time_ns() - wall_time)

    msg = @sprintf("Time: %s, n: %d, Δt: %s, max|u|: (%.2e, %.2e, %.2e) m s⁻¹, \
                   extrema(T): (%.2f, %.2f) ᵒC, wall time: %s \n",
                   prettytime(sim), iteration(sim), prettytime(sim.Δt),
                   umax..., Tmax, Tmin, prettytime(step_time))

    ClimaOcean.@root @info(msg)

    wall_time = time_ns()

    return nothing
end

function mtn₁(λ, φ)
    λ₁ = 70
    φ₁ = 55
    dφ = 5
    return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
end

function mtn₂(λ, φ)
    λ₁ = 70
    λ₂ = λ₁ + 180
    φ₂ = 55
    dφ = 5
    return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
end

# Simple initial condition for producing pretty pictures
function smooth_step(φ)
    φ₀ = 40
    dφ = 5
    return (1 - tanh((abs(φ) - φ₀) / dφ)) / 2
end

function Tᵢ(λ, φ, z)
    dTdz = 1e-3
    return (30 + dTdz * z) * smooth_step(φ) + rand()
end

function Sᵢ(λ, φ, z)
    dSdz = - 5e-3
    return dSdz * z + rand()
end

zonal_wind(λ, φ) = 4 * sind(2φ)^2 - 2 * exp(-(abs(φ) - 12)^2 / 72)
sunlight(λ, φ) = -200 - 600 * cosd(φ)^2
Tatm(λ, φ, z=0) = 30 * cosd(φ)

function gaussian_islands_tripolar_grid(arch::Architectures.AbstractArchitecture)
    # Horizontal resolution
    resolution = 2 # 1/4 for quarter degree
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 170 / resolution)

    # Vertical resolution
    Nz = 20 # eventually we want to increase this to between 100-600

    # Time step. This must be decreased as resolution is decreased.
    Δt = 1minutes

    # Grid setup
    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces)

    #underlying_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces,
    #                                        longitude=(0, 360), latitude=(-80, 80))
    zb = z_faces[1]
    h = -zb + 100
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))

    return @gbprofile "ImmersedBoundaryGrid" ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(gaussian_islands))
end

function data_free_ocean_climate_simulation_init(arch::Architectures.AbstractArchitecture=Architectures.ReactantState())

    grid = gaussian_islands_tripolar_grid(arch)

    # See visualize_ocean_climate_simulation.jl for information about how to
    # visualize the results of this run.
    ocean = @gbprofile "ocean_simulation" ocean_simulation(grid)

    @gbprofile "set_ocean_model" set!(ocean.model, T=Tᵢ, S=Sᵢ)

    # Set up an atmosphere
    atmos_times = range(0, 1days, length=24)

    atmos_grid = LatitudeLongitudeGrid(arch,
                                       size = (360, 180),
                                       longitude = (0, 360),
                                       latitude = (-90, 90),
                                       topology = (Periodic, Bounded, Flat))

    atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times)

    Ta = Field{Center, Center, Nothing}(atmos_grid)
    ua = Field{Center, Center, Nothing}(atmos_grid)
    Qs = Field{Center, Center, Nothing}(atmos_grid)

    set!(Ta, Tatm)
    set!(ua, zonal_wind)
    set!(Qs, sunlight)

    parent(atmosphere.tracers.T) .= parent(Ta) .+ 273.15
    parent(atmosphere.velocities.u) .= parent(ua)
    parent(atmosphere.tracers.q) .= 0
    parent(atmosphere.downwelling_radiation.shortwave) .= parent(Qs)

    # Atmospheric model
    radiation  = Radiation(arch)

    # Coupled model and simulation
    solver_stop_criteria = FixedIterations(5) # note: more iterations = more accurate
    atmosphere_ocean_flux_formulation = SimilarityTheoryFluxes(; solver_stop_criteria)
    interfaces = ComponentInterfaces(atmosphere, ocean; radiation, atmosphere_ocean_flux_formulation)
    coupled_model = @gbprofile "OceanSeaIceModel" OceanSeaIceModel(ocean; atmosphere, radiation, interfaces)
    simulation = @gbprofile "Simulation" Simulation(coupled_model; Δt=20minutes, stop_iteration=40)
    pop!(simulation.callbacks, :nan_checker)

    wall_time = time_ns()

    if !(arch isa Architectures.ReactantState)
        add_callback!(simulation, progress, IterationInterval(10))
    end

    # Output
    prefix = if arch isa Distributed
        "ocean_climate_simulation_rank$(arch.local_rank)"
    else
        "ocean_climate_simulation_serial"
    end

    Nz = size(grid, 3)
    outputs = merge(ocean.model.velocities, ocean.model.tracers)
    if !(arch isa Architectures.ReactantState)
        surface_writer = JLD2OutputWriter(ocean.model, outputs,
        				  filename = prefix * "_surface.jld2",
        				  indices = (:, :, Nz),
        				  schedule = TimeInterval(3days),
        				  overwrite_existing = true)

        simulation.output_writers[:surface] = surface_writer
    end

    return simulation

end # data_free_ocean_climate_simulation_init
