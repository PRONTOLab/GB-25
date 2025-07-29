using Oceananigans
using Oceananigans.Architectures: ReactantState
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using ClimaOcean.Diagnostics: MixedLayerDepthField

using Oceananigans.Grids: λnode, φnode, znode, new_data

using Oceananigans.Fields: instantiated_location, location, scan_indices, indices, FieldStatus, compute_at!, get_neutral_mask, interior, condition_operand

# https://github.com/CliMA/Oceananigans.jl/blob/c29939097a8d2f42966e930f2f2605803bf5d44c/src/AbstractOperations/binary_operations.jl#L5
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{Oceananigans.AbstractOperations.BinaryOperation{LX, LY, LZ, O, A, B, IA, IB, G, T}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {LX, LY, LZ, O, A, B, IA, IB, G, T}
    LX2 = Reactant.traced_type_inner(LX, seen, mode, track_numbers, sharding, runtime)
    LY2 = Reactant.traced_type_inner(LY, seen, mode, track_numbers, sharding, runtime)
    LZ2 = Reactant.traced_type_inner(LZ, seen, mode, track_numbers, sharding, runtime)

    O2 = Reactant.traced_type_inner(O, seen, mode, track_numbers, sharding, runtime)

    A2 = Reactant.traced_type_inner(A, seen, mode, track_numbers, sharding, runtime)
    B2 = Reactant.traced_type_inner(B, seen, mode, track_numbers, sharding, runtime)
    IA2 = Reactant.traced_type_inner(IA, seen, mode, track_numbers, sharding, runtime)
    IB2 = Reactant.traced_type_inner(IB, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)

    T2 = eltype(G2)
    return Oceananigans.AbstractOperations.BinaryOperation{LX2, LY2, LZ2, O2, A2, B2, IA2, IB2, G2, T2}
end

using SeawaterPolynomials

using Enzyme

throw_error = true
include_halos = true
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

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
    buoyancy = SeawaterBuoyancy(equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType))

    # Closures:
    # diffusivity scheme we need for GM/Redi.
    # κ_symmetric is the parameter we need to train for (can be scalar or a spatial field),
    # κ_skew is GM parameter (also scalar or spatial field),
    # we might want to use the Isopycnal Tensor instead of small slope (small slope common),
    # unsure of slope limiter and time disc.
    redi_diffusivity = IsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), Float64;
                                                        κ_skew = 0.0,
                                                        κ_symmetric = 0.0)
                                                        #isopycnal_tensor = IsopycnalTensor(),
                                                        #slope_limiter = FluxTapering(1e-2))

    horizontal_closure = HorizontalScalarDiffusivity(ν = 5000.0, κ = 1000.0)
    #vertical_closure   = VerticalScalarDiffusivity(ν = 1e-2, κ = 1e-5) 
    vertical_closure   = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    #vertical_closure = Oceananigans.TurbulenceClosures.TKEDissipationVerticalDiffusivity()
    closure = (redi_diffusivity, vertical_closure)

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis()

    tracers = (:T, :S, :e)

    grid = simple_latitude_longitude_grid(arch, Nx, Ny, Nz)

    momentum_advection = VectorInvariant() #WENOVectorInvariant(order=5)
    tracer_advection   = Centered(order=2) #WENO(order=5)
    
    #
    # Temperature Relaxation Enforced as a Flux BC:
    #
    ρₒ = 1026.0 # kg m⁻³, average density at the surface of the world ocean
    cₚ = 3994   # specific heat, J / kg * K
    Lφ =  30
    φ₀ = -60
    τ  = 864000 # Relaxation timescale, equal to 10 days

    # TODO: replace with discrete form
    surface_condition_T(i, j, grid, clock, model_fields) = (ρₒ/τ) * (model_fields.T[i, j, Nz] - (-2 + 12(φnode(j, grid, Center()) - φ₀) / Lφ))
    T_top_bc = FluxBoundaryCondition(surface_condition_T, discrete_form=true)
    
    north_condition_T(i, k, grid, clock, model_fields) = (ρₒ/τ) * (model_fields.T[i, Ny, k] - (-2 + 12(-30 - φ₀) * exp(znode(k, grid, Center())/800) / Lφ))
    T_north_bc = FluxBoundaryCondition(north_condition_T, discrete_form=true)

    T_bcs = FieldBoundaryConditions(north=T_north_bc, top=T_top_bc)
    
    #
    # Momentum BCs:
    #
    no_slip_bc = ValueBoundaryCondition(Field{Face, Center, Nothing}(grid))
    u_top_bc   = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    u_bcs = FieldBoundaryConditions(north=no_slip_bc, south=no_slip_bc, top=u_top_bc)

    boundary_conditions = (u=u_bcs, T=T_bcs)

    model = HydrostaticFreeSurfaceModel(; grid,
                                          free_surface = free_surface,
                                          closure = closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          coriolis = coriolis,
                                          momentum_advection = momentum_advection,
                                          tracer_advection = tracer_advection,
                                          boundary_conditions = boundary_conditions)

    set!(model.tracers.e, 1e-6)
    model.clock.last_Δt = Δt

    return model
end

Nx = 362 #128
Ny = 32
Nz = 30

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, Nx, Ny, Nz, 1200)

@allowscalar @show rmodel.grid

@allowscalar scan = Integral(rmodel.velocities.u, dims=(2,3))

operand = scan.operand
grid = operand.grid
LX, LY, LZ = loc = instantiated_location(scan)
thing = scan_indices(scan.type, indices(operand); dims=scan.dims)

data = new_data(grid, loc, thing)
recompute_safely = false

boundary_conditions = FieldBoundaryConditions(grid, loc, thing)
status = recompute_safely ? nothing : FieldStatus()

@allowscalar rzonal_transport = Field(loc, grid, data, boundary_conditions, thing, scan, status)

#=
s = rzonal_transport.operand
compute_at!(s.operand, nothing)

@allowscalar @show s.operand

@allowscalar @show condition_operand(s.operand, nothing, 0)

thing_operand = rmodel.velocities.u * rmodel.velocities.u

new_thing = Base.initarray!(interior(rzonal_transport), identity, Base.add_sum, true, thing_operand)

@show identity
@show Base.add_sum
@show new_thing
@show thing_operand
=#
#@allowscalar Reactant.compile(Reactant.mymapreducedim!, (identity, Base.add_sum, new_thing, thing_operand))

#@allowscalar Base.mapreducedim!(identity, Base.add_sum, new_thing, thing_operand)

#@allowscalar sum!(identity, interior(rzonal_transport), thing_operand)

#@allowscalar sum!(identity, interior(rzonal_transport), condition_operand(s.operand, nothing, 0))

#@allowscalar sum!(rzonal_transport, s.operand)

#compute!(rzonal_transport)

@allowscalar rzonal_transport = Field(Integral(rmodel.velocities.u))