using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using SeawaterPolynomials
using Reactant
using Random
using Printf

function print_comparison(name, ψ1, ψ2)
    ψ1 = Array(parent(ψ1))
    ψ2 = Array(parent(ψ2))
    δ = ψ1 .- ψ2
    @printf("(%4s) ψ₁ ≈ ψ₂: %-5s, max|ψ₁|, max|ψ₂|: %.15e, %.15e, max|δ|: %.15e \n",
            name, ψ1 ≈ ψ2, maximum(abs, ψ1), maximum(abs, ψ2), maximum(abs, δ))
end


function compare_states(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        print_comparison(name, Ψ1[name], Ψ2[name])

        if !(name ∈ (:w, :η))
            Gⁿ1 = m1.timestepper.Gⁿ
            Gⁿ2 = m2.timestepper.Gⁿ
            print_comparison("Gⁿ.$name", Gⁿ1[name], Gⁿ2[name]) 

            G⁻1 = m1.timestepper.G⁻
            G⁻2 = m2.timestepper.G⁻
            print_comparison("G⁻.$name", G⁻1[name], G⁻2[name]) 
        end
    end
    return nothing
end

function sync_states!(m1, m2)
    Ψ1 = Oceananigans.fields(m1)
    Ψ2 = Oceananigans.fields(m2)
    for name in keys(Ψ1)
        ψ1p = parent(Ψ1[name])
        ψ2p = parent(Ψ2[name])
        copyto!(ψ1p, Reactant.to_rarray(ψ2p))
    end
    return nothing
end


function simple_latitude_longitude_grid(arch, resolution, Nz)
    Nx = convert(Int, 360 / resolution)
    Ny = convert(Int, 160 / resolution)

    z_faces = exponential_z_faces(; Nz, depth=4000, h=30) # may need changing for very large Nz

    grid = LatitudeLongitudeGrid(arch, size=(Nx, Ny, Nz), halo=(7, 7, 7), z=z_faces,
        latitude = (-80, 80),
        longitude = (0, 360)
    )

    return grid
end

@inline function bᵢ(λ, φ, z)
    N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
    Δb = 0.005 # [m/s²] buoyancy difference
    φ₀ = 50
    Δφ = 20
    γ = π/2 - 2π * (φ₀ - φ) / Δφ
    μ = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return N² * z + Δb * μ + 1e-2 * Δb * randn()
end

function baroclinic_instability_model(arch; resolution, Δt, Nz,
    grid = :simple_lat_lon,
    
    # Fewer substeps can be used at higher resolutions
    free_surface = SplitExplicitFreeSurface(substeps=30),

    # TEOS10 is a 54-term polynomial that relates temperature (T),
    # and salinity (S) to buoyancy
    buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState()),

    # CATKE correctness is not established yet, so we are using a simpler closure
    # closure = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), κ=1e-5, ν=1e-4),

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis(),

    # Simple momentum advection schemes. May need to be reconsidered
    # due to Float32.
    momentum_advection = WENOVectorInvariant(order=5),
    tracer_advection = WENO(order=5),
    )

    if buoyancy isa BuoyancyTracer
        tracers = [:b]
    elseif buoyancy isa SeawaterBuoyancy
        tracers = [:T, :S]
    else
        tracers = []
    end

    if closure isa Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity
        push!(tracers, :e)
    elseif closure isa Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity
        push!(tracers, :e)
        push!(tracers, :ϵ)
    end

    tracers = tuple(tracers...)

    if grid === :gaussian_islands
        grid = gaussian_islands_tripolar_grid(arch, resolution, Nz)
    elseif grid === :simple_lat_lon
        grid = simple_latitude_longitude_grid(arch, resolution, Nz)
    end

    model = HydrostaticFreeSurfaceModel(;
        grid, free_surface, closure, buoyancy, tracers,
        coriolis, momentum_advection, tracer_advection,
    )

    Random.seed!(42)

    if buoyancy isa SeawaterBuoyancy
        set!(model, T=Tᵢ, S=Sᵢ)
    elseif buoyancy isa BuoyancyTracer
        set!(model, b=bᵢ)
    end

    model.clock.last_Δt = Δt

    return model
end

