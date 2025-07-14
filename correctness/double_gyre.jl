using Oceananigans
using Oceananigans.Architectures: ReactantState
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using ClimaOcean.Diagnostics: MixedLayerDepthField


using SeawaterPolynomials

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
    # Momentum BCs:
    #
    no_slip_bc = ValueBoundaryCondition(Field{Face, Center, Nothing}(grid))
    u_top_bc   = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    u_bcs = FieldBoundaryConditions(north=no_slip_bc, south=no_slip_bc, top=u_top_bc)
    v_bcs = FieldBoundaryConditions(east=no_slip_bc, west=no_slip_bc)

    boundary_conditions = (u=u_bcs, )

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

Nx = 362
Ny = 32
Nz = 30

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, Nx, Ny, Nz, 1200)

@info "Compiling..."

@show rmodel.grid

mld  = MixedLayerDepthField(rmodel.buoyancy, rmodel.grid, rmodel.tracers)

tic = time()
rcompute! = @compile raise_first=true raise=true sync=true compute!(mld)
compile_toc = time() - tic

@show compile_toc

rcompute!(mld)
