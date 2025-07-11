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

using Enzyme

throw_error = true
include_halos = true
rtol = sqrt(eps(Float64))
atol = sqrt(eps(Float64))

function set_tracers(grid;
                     Lφ::Real = 30, # Meridional length in degrees
                     φ₀::Real = -60 # Degrees north of equator for the southern edge)
                    )
    fₜ(λ, φ, z) = -2 + 12(φ - φ₀) * exp(z/800) / Lφ # passable, linear in y
    #fₜ(λ, φ, z) = -2 + 12exp(z/800 + (φ₀ + φ)) experiment if it should be exponential in y
    fₛ(λ, φ, z) = 0 #35 # This example does not use salinity

    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)

    @allowscalar set!(Tᵢ, fₜ)
    @allowscalar set!(Sᵢ, fₛ)

    return Tᵢ, Sᵢ
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

function wind_stress_init(grid;
                            ρₒ::Real = 1026.0, # kg m⁻³, average density at the surface of the world ocean
                            Lφ::Real = 30, # Meridional length in degrees
                            φ₀::Real = -60.0 # Degrees north of equator for the southern edge
                            )
    wind_stress = Field{Face, Center, Nothing}(grid)

    τ₀ = 0.2 # N m⁻² / density of seawater
    @inline τx(λ, φ) = (τ₀ / ρₒ) * sin(π * (φ - φ₀) / (Lφ))

    set!(wind_stress, τx)
    return wind_stress
end

function loop!(model)
    Δt = model.clock.last_Δt + 0
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace checkpointing = true track_numbers = false for i = 1:100
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
    model.clock.last_Δt = 1200

    # Step it forward
    loop!(model)

    return nothing
end

function estimate_tracer_error(model, initial_temperature, initial_salinity, wind_stress, mld)
    time_step_double_gyre!(model, initial_temperature, initial_salinity, wind_stress)
    # Compute the mean mixed layer depth:
    compute!(mld)
    Nλ, Nφ, _ = size(model.grid)
    #=
    avg_mld = 0.0
    
    for j0 = 1:Nφ, i0 = 1:Nλ
        @allowscalar avg_mld += @inbounds model.velocities.u[i0, j0, 1]^2
    end
    avg_mld = avg_mld / (Nλ * Nφ)
    =#
    # Hard way
    c² = parent(model.velocities.u).^2
    avg_mld = sum(c²)
    #@allowscalar avg_mld = model.velocities.u[10, 10, 1] #sum(c²)
    
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

Nx = 362
Ny = 32
Nz = 30

Oceananigans.defaults.FloatType = Float64

@info "Generating model..."
rarch = ReactantState()
rmodel = double_gyre_model(rarch, Nx, Ny, Nz, 1200)

@info rmodel.buoyancy

rTᵢ, rSᵢ     = set_tracers(rmodel.grid)
rwind_stress = wind_stress_init(rmodel.grid)

@info "Compiling..."

dmodel = Enzyme.make_zero(rmodel)
dTᵢ = Field{Center, Center, Center}(rmodel.grid)
dSᵢ = Field{Center, Center, Center}(rmodel.grid)
dJ  = Field{Face, Center, Nothing}(rmodel.grid)

@info dmodel
@info dmodel.closure

@show rmodel.grid

mld  = MixedLayerDepthField(rmodel.buoyancy, rmodel.grid, rmodel.tracers)
dmld = MixedLayerDepthField(dmodel.buoyancy, dmodel.grid, dmodel.tracers)

#=
using GLMakie
# Build init temperature fields:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
T = rTᵢ

fig, ax, hm = heatmap(view(T, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("init_T_surface.png", fig)

fig, ax, hm = heatmap(view(T, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=-4000, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("init_T_bottom.png", fig)

# Energy:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
e = rmodel.tracers.e

fig, ax, hm = heatmap(view(e, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("init_e_surface.png", fig)

fig, ax, hm = heatmap(view(e, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=-4000, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("init_e_bottom.png", fig)

# Wind stress:
x, y, z = nodes(rmodel.grid, (Face(), Center(), Nothing()))
fig, ax, hm = heatmap(view(rwind_stress, :, :),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "zonal wind_stress(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save("init_wind_stress.png", fig)

# As sanity checks we'll also plot initial gradients for each (should all be 0):
# Build init temperature fields:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
T = dTᵢ

fig, ax, hm = heatmap(view(T, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "dT(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("init_dT_surface.png", fig)

fig, ax, hm = heatmap(view(T, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "dT(x, y, z=0, t=-4000)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("init_dT_bottom.png", fig)

# Energy:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
e = dmodel.tracers.e

fig, ax, hm = heatmap(view(e, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "de(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("init_de_surface.png", fig)

fig, ax, hm = heatmap(view(e, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "de(x, y, z=0, t=-4000)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("init_de_bottom.png", fig)

# Wind stress:
x, y, z = nodes(rmodel.grid, (Face(), Center(), Nothing()))

fig, ax, hm = heatmap(view(dJ, :, :),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "dwind_stress(x, y, z=0, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save("init_dwind_stress.png", fig)

# Mixed layer depth:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Nothing()))

fig, ax, hm = heatmap(view(mld, :, :),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "mld(x, y, t=0)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m]")

save("init_mld.png", fig)

=#
tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld)
#rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true differentiate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld,
#                                                                                                        dmodel, dTᵢ, dSᵢ, dJ, dmld)
compile_toc = time() - tic

@show compile_toc

restimate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld)

#dedν, dJ = rdifferentiate_tracer_error(rmodel, rTᵢ, rSᵢ, rwind_stress, mld, dmodel, dTᵢ, dSᵢ, dJ, dmld)

#=
Add plots of gradient fields here, want to do:

1. Wind stress
2. temperature
3. salinity
4. CATKE parameters

=#
#=
# First gradient data:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
T = dTᵢ

fig, ax, hm = heatmap(view(T, :, :, 30),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "dT(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("final_dT_surface.png", fig)

fig, ax, hm = heatmap(view(T, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "dT(x, y, z=-4000, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("final_dT_bottom.png", fig)

# Energy:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
e = dmodel.tracers.e

fig, ax, hm = heatmap(view(e, :, :, 30),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("final_de_surface.png", fig)

fig, ax, hm = heatmap(view(e, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=-4000, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("final_de_bottom.png", fig)

# Wind stress:
x, y, z = nodes(rmodel.grid, (Face(), Center(), Nothing()))

fig, ax, hm = heatmap(view(dJ, :, :),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "dwind_stress(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save("final_dwind_stress.png", fig)

# Build final temperature fields:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
T = rmodel.tracers.T

fig, ax, hm = heatmap(view(T, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("final_T_surface.png", fig)

fig, ax, hm = heatmap(view(T, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "T(x, y, z=-4000, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[degrees C]")

save("final_T_bottom.png", fig)

# Energy:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Center()))
e = rmodel.tracers.e

fig, ax, hm = heatmap(view(e, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("final_e_surface.png", fig)

fig, ax, hm = heatmap(view(e, :, :, 1),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "e(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[energy]")

save("final_e_bottom.png", fig)

# Zonal velocity:
x, y, z = nodes(rmodel.grid, (Face(), Center(), Center()))

fig, ax, hm = heatmap(view(rmodel.velocities.u, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "u(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")

save("final_surface_u.png", fig)
#=
# Meridional velocity:
x, y, z = nodes(rmodel.grid, (Center(), Face(), Center()))

fig, ax, hm = heatmap(view(rmodel.velocities.v, :, :, Nz),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "v(x, y, z=0, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m/s]")
=#
save("final_surface_v.png", fig)

# Mixed layer depth:
x, y, z = nodes(rmodel.grid, (Center(), Center(), Nothing()))

fig, ax, hm = heatmap(view(mld, :, :),
                      colormap = :deep,
                      axis = (xlabel = "x [degrees]",
                              ylabel = "y [degrees]",
                              title = "mld(x, y, t=400min)",
                              titlesize = 24))

Colorbar(fig[1, 2], hm, label = "[m]")

save("final_mld.png", fig)

i = 10
j = 10

@allowscalar @show dJ[i, j]


# Produce finite-difference gradients for comparison:
ϵ_list = [1e-1, 1e-2, 1e-3, 1e-4] #, 1e-5, 1e-6, 1e-7, 1e-8]

@allowscalar gradient_list = Array{Float64}[]

for ϵ in ϵ_list
    rmodelP = double_gyre_model(rarch, Nx, Ny, Nz, 1200)
    rTᵢP, rSᵢP      = set_tracers(rmodelP.grid)
    rwind_stressP = wind_stress_init(rmodelP.grid)

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, j])

    @allowscalar rwind_stressP[i, j] = rwind_stressP[i, j] + ϵ * abs(rwind_stressP[i, j])

    sq_surface_uP = restimate_tracer_error(rmodelP, rTᵢP, rSᵢP, rwind_stressP, mld)

    rmodelM = double_gyre_model(rarch, Nx, Ny, Nz, 1200)
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