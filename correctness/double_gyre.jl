using Oceananigans
using Oceananigans.Architectures: ReactantState
using ClimaOcean
using Reactant
using GordonBell25
#Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
#Reactant.allowscalar(true)
Reactant.set_default_backend("cpu")

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, SmallSlopeIsopycnalTensor, FluxTapering, CATKEVerticalDiffusivity
using Oceananigans.BoundaryConditions: NoFluxBoundaryCondition
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using ClimaOcean.Diagnostics: MixedLayerDepthField

using Oceananigans.Grids: λnode, φnode, znode


using SeawaterPolynomials

using Enzyme

throw_error = true
include_halos = false
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

function set_tracers(grid;
                     Lφ::Real = 30, # Meridional length in degrees
                     φ₀::Real = -60 # Degrees north of equator for the southern edge)
                    )
    fₜ(λ, φ, z) = -2 + 12(φ - φ₀) * exp(z/800) / Lφ # passable, linear in y
    #fₜ(λ, φ, z) = -2 + 12exp(z/800 + (φ₀ + φ)) experiment if it should be exponential in y
    fₛ(λ, φ, z) = 30

    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)

    @allowscalar set!(Tᵢ, fₜ)
    @allowscalar set!(Sᵢ, fₛ)

    return Tᵢ, Sᵢ
end

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

#####
##### Utilities for bottom drag:
#####

@inline ϕ²(i, j, k, grid, ϕ)    = @inbounds ϕ[i, j, k]^2
@inline spᶠᶜᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, ϕ², Φ.v))
@inline spᶜᶠᶜ(i, j, k, grid, Φ) = @inbounds sqrt(Φ.v[i, j, k]^2 + ℑxyᶜᶠᵃ(i, j, k, grid, ϕ², Φ.u))

@inline u_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.u[i, j, 1] * spᶠᶜᶜ(i, j, 1, grid, Φ)
@inline v_quadratic_bottom_drag(i, j, grid, c, Φ, μ) = @inbounds - μ * Φ.v[i, j, 1] * spᶜᶠᶜ(i, j, 1, grid, Φ)

# Keep a constant linear drag parameter independent on vertical level
@inline u_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, k] * spᶠᶜᶜ(i, j, k, grid, fields)
@inline v_immersed_bottom_drag(i, j, k, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, k] * spᶜᶠᶜ(i, j, k, grid, fields)

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
    redi_diffusivity   = IsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), Float64;
                                                           κ_skew = 1000,
                                                           κ_symmetric = 1000,
                                                           isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                                           slope_limiter = FluxTapering(1e-2))

    horizontal_closure = HorizontalScalarDiffusivity(ν = 10000, κ = 100)
    vertical_closure   = Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivity()
    closure            = (redi_diffusivity, horizontal_closure, vertical_closure)

    # Coriolis forces for a rotating Earth
    coriolis = HydrostaticSphericalCoriolis()

    tracers = (:T, :S, :e)

    underlying_grid = simple_latitude_longitude_grid(arch, Nx, Ny, Nz)

    ridge(λ, φ) = 4100exp(-0.005(λ - 120)^2) * (1 / (1 + exp(-3(φ+45))) + 1 / (1 + exp(-3(-φ-55)))) - 4000 # might be needed
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))

    momentum_advection = WENOVectorInvariant()
    tracer_advection   = WENO() #Centered(order=2)
    
    #
    # Temperature Relaxation Enforced as a Flux BC:
    #
    ρₒ = 1026.0 # kg m⁻³, average density at the surface of the world ocean
    cₚ = 3994   # specific heat, J / kg * K
    Lφ =  30
    φ₀ = -60
    τ  = 864000 # Relaxation timescale, equal to 10 days

    # TODO: replace with discrete form
    surface_condition_T(i, j, grid, clock, model_fields) = (80 / τ) * (model_fields.T[i, j, Nz] - (-2 + 12(φnode(j, grid, Center()) - φ₀) / Lφ))
    T_top_bc = FluxBoundaryCondition(surface_condition_T, discrete_form=true)
    
    north_condition_T(i, k, grid, clock, model_fields) = (110000 / τ) * (model_fields.T[i, Ny, k] - (-2 + 12(-30 - φ₀) * exp(znode(k, grid, Center())/800) / Lφ))
    T_north_bc = FluxBoundaryCondition(north_condition_T, discrete_form=true)

    T_bcs = FieldBoundaryConditions(top=T_top_bc) # north=T_north_bc
    
    #
    # Momentum BCs:
    #
    no_slip_bc = ValueBoundaryCondition(0.0)
    u_top_bc   = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))

    #
    # Bottom drag:
    #
    bottom_drag_coefficient = 0.003 # was 0.003

    u_immersed_drag = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
    v_immersed_drag = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

    u_immersed_bc = ImmersedBoundaryCondition(bottom=u_immersed_drag)
    v_immersed_bc = ImmersedBoundaryCondition(bottom=v_immersed_drag)

    u_bot_bc = FluxBoundaryCondition(u_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
    v_bot_bc = FluxBoundaryCondition(v_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)


    u_bcs = FieldBoundaryConditions(north=no_slip_bc, south=no_slip_bc, top=u_top_bc, bottom=u_bot_bc, immersed=u_immersed_bc)
    v_bcs = FieldBoundaryConditions(bottom=v_bot_bc, immersed=v_immersed_bc)

    boundary_conditions = (u=u_bcs, T=T_bcs, v=v_bcs)

    #
    # Forcings, to get Relaxation in Northern sponge layer
    #
    #northern_sponge_u(i, j, k, grid, clock, model_fields) = -(1 / τ) * model_fields.u[i, j, k] * ((j==Ny) + 0.25(j==(Ny-1)))
    #northern_sponge_v(i, j, k, grid, clock, model_fields) = -(1 / τ) * model_fields.v[i, j, k] * ((j==Ny) + 0.25(j==(Ny-1)))
    northern_sponge_T(i, j, k, grid, clock, model_fields) = -(1 / τ) * (model_fields.T[i, Ny, k] - (-2 + 12(-30 - φ₀) * exp(znode(k, grid, Center())/800) / Lφ)) * ((j==Ny) + 0.25(j==(Ny-1)))

    #u_forcing = Forcing(northern_sponge_u, discrete_form=true)
    #v_forcing = Forcing(northern_sponge_v, discrete_form=true)
    T_forcing = Forcing(northern_sponge_T, discrete_form=true)

    forcings = (T=T_forcing,)

    model = HydrostaticFreeSurfaceModel(; grid,
                                          free_surface = free_surface,
                                          closure = closure,
                                          buoyancy = buoyancy,
                                          tracers = tracers,
                                          coriolis = coriolis,
                                          momentum_advection = momentum_advection,
                                          tracer_advection = tracer_advection,
                                          boundary_conditions = boundary_conditions,
                                          forcing = forcings)

    #set!(model.tracers.e, 1e-6)
    model.clock.last_Δt = Δt

    return model, underlying_grid
end

function wind_stress_init(grid;
                            ρₒ::Real = 1026.0, # kg m⁻³, average density at the surface of the world ocean
                            Lφ::Real = 30, # Meridional length in degrees
                            φ₀::Real = -60.0 # Degrees north of equator for the southern edge
                            )
    wind_stress = Field{Face, Center, Nothing}(grid)

    τ₀ = 0.2 / ρₒ # N m⁻² / density of seawater
    @inline τx(λ, φ) = -τ₀ * sin(π * (φ - φ₀) / (Lφ))

    set!(wind_stress, τx)
    return wind_stress
end

function loop!(model)
    Δt = model.clock.last_Δt + 0
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace mincut = true track_numbers = false for i = 1:9999
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

function time_step_double_gyre!(model, Tᵢ, Sᵢ, wind_stress)

    set!(model.tracers.T, Tᵢ)
    set!(model.tracers.S, Sᵢ)
    set!(model.velocities.u.boundary_conditions.top.condition, wind_stress)

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0

    # Step it forward
    loop!(model)

    return nothing
end

function estimate_tracer_error(model, initial_temperature, initial_salinity, wind_stress, mld)
    time_step_double_gyre!(model, initial_temperature, initial_salinity, wind_stress)
    # Compute the mean mixed layer depth:
    #compute!(mld)
    Nλ, Nφ, Nz = size(model.grid)
    #=
    avg_mld = 0.0
    
    for j0 = 1:Nφ, i0 = 1:Nλ
        @allowscalar avg_mld += @inbounds model.velocities.u[i0, j0, 1]^2
    end
    avg_mld = avg_mld / (Nλ * Nφ)
    =#
    # Hard way
    c² = parent(model.tracers.T).^2
    avg_mld = sum(c²) / (Nλ * Nφ * Nz)
    return avg_mld
end

function differentiate_tracer_error(model, Tᵢ, Sᵢ, J, mld, dmodel, dTᵢ, dSᵢ, dJ, dmld)

    dedν = autodiff(set_strong_zero(Enzyme.Reverse),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(Sᵢ, dSᵢ),
                    Duplicated(J, dJ),
                    Duplicated(mld, dmld))

    return dedν, dJ
end

Nx = 362 #128
Ny = 32
Nz = 50
time_step = 600

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel, runderlying_grid = double_gyre_model(rarch, Nx, Ny, Nz, time_step)

@info rmodel.buoyancy

rTᵢ, rSᵢ     = set_tracers(rmodel.grid)
rwind_stress = wind_stress_init(rmodel.grid)

@info "Generating Gradients"

dmodel = Enzyme.make_zero(rmodel)
dTᵢ = Field{Center, Center, Center}(rmodel.grid)
dSᵢ = Field{Center, Center, Center}(rmodel.grid)
dJ  = Field{Face, Center, Nothing}(rmodel.grid)

@info rmodel
@info rmodel.closure

mld  = MixedLayerDepthField(rmodel.buoyancy, rmodel.grid, rmodel.tracers)
dmld = MixedLayerDepthField(dmodel.buoyancy, dmodel.grid, dmodel.tracers)

set!(rmodel.tracers.T, rTᵢ)

#@show argmax(T)
#@show T[1,32,30]

#
# Plotting:
#
graph_directory = "run_steps10000_timestep600_salinity30_windstressNeg02_ridgeFull_relaxationS80N111K_spongeNT_e0_Nz50_horizontalvisc10000_horizontaldiff100_ridgeWidthX50_ridgeSmoothed_quadraticBottomDrag_WENO/"

outputs = (u=rmodel.velocities.u, v=rmodel.velocities.v, T=rmodel.tracers.T, e=rmodel.tracers.e, SSH=rmodel.free_surface.η)

using FileIO, JLD2

Base.Filesystem.mkdir(graph_directory)

filename = graph_directory * "data_init.jld2"

jldsave(filename; Nx, Ny, Nz,
                  bottom_height=convert(Array, interior(rmodel.grid.immersed_boundary.bottom_height)),
                  T_init=convert(Array, interior(rmodel.tracers.T)),
                  e_init=convert(Array, interior(rmodel.tracers.e)),
                  wind_stress=convert(Array, interior(rwind_stress)))

@info "Compiling..."
tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld)
#rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true differentiate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld,
#                                                                                                        dmodel, dTᵢ, dSᵢ, dJ, dmld)
compile_toc = time() - tic

@show compile_toc

@info "Running..."

tic = time()
restimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld)
#dedν, dJ = rdifferentiate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld, dmodel, dTᵢ, dSᵢ, dJ, dmld)
rrun_toc = time() - tic
@show rrun_toc

filename = graph_directory * "data_final.jld2"

jldsave(filename; Nx, Ny, Nz,
                  T_final=convert(Array, interior(rmodel.tracers.T)),
                  e_final=convert(Array, interior(rmodel.tracers.e)),
                  ssh=convert(Array, interior(rmodel.free_surface.η)),
                  u=convert(Array, interior(rmodel.velocities.u)),
                  v=convert(Array, interior(rmodel.velocities.v)))


i = 10
j = 10

#@allowscalar @show dJ[i, j]

#=
# Produce finite-difference gradients for comparison:
ϵ_list = [1e-1, 1e-2, 1e-3, 1e-4] #, 1e-5, 1e-6, 1e-7, 1e-8]

@allowscalar gradient_list = Array{Float64}[]

for ϵ in ϵ_list
    rmodelP, _ = double_gyre_model(rarch, Nx, Ny, Nz, time_step)
    rTᵢP, rSᵢP      = set_tracers(rmodelP.grid)
    rwind_stressP = wind_stress_init(rmodelP.grid)

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, j])

    @allowscalar rwind_stressP[i, j] = rwind_stressP[i, j] + ϵ * abs(rwind_stressP[i, j])

    sq_surface_uP = restimate_tracer_error(rmodelP, rTᵢP, rSᵢP, rwind_stressP, mld)

    rmodelM, _ = double_gyre_model(rarch, Nx, Ny, Nz, time_step)
    rTᵢM, rSᵢM      = set_tracers(rmodelM.grid)
    rwind_stressM = wind_stress_init(rmodelM.grid)
    @allowscalar rwind_stressM[i, j] = rwind_stressM[i, j] - ϵ * abs(rwind_stressM[i, j])

    sq_surface_uM = restimate_tracer_error(rmodelM, rTᵢM, rSᵢM, rwind_stressM, mld)

    dsq_surface_u = (sq_surface_uP - sq_surface_uM) / diff

    #push!(gradient_list, dsq_surface_u)
    @show ϵ, dsq_surface_u

end

@info gradient_list
=#