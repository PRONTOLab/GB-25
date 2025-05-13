using GordonBell25
using Oceananigans
using Reactant

function fake_time_step_catke_equation!(model)

    closure = model.closure
    diffusivity_fields = model.diffusivity_fields
    arch = model.architecture
    grid = model.grid
    Gⁿe = model.timestepper.Gⁿ.e
    G⁻e = model.timestepper.G⁻.e
    κe = diffusivity_fields.κe
    Le = diffusivity_fields.Le
    previous_velocities = diffusivity_fields.previous_velocities
    Δt = model.clock.last_Δt
    χ = model.timestepper.χ

    Oceananigans.Utils.launch!(
        arch, grid, :xyz,
        Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities.substep_turbulent_kinetic_energy!,
        κe, Le, grid, closure,
        model.velocities, previous_velocities, # try this soon: model.velocities, model.velocities,
        model.tracers, model.buoyancy, diffusivity_fields,
        Δt, χ, Gⁿe, G⁻e)

    return nothing
end
    
throw_error = false
include_halos = true
rtol = sqrt(eps(Float64))
atol = 0

model_kw = (
    halo = (8, 8, 8),
    closure = CATKEVerticalDiffusivity(),
    Δt = 1e-9,
)

Nx, Ny, Nz = 112, 112, 16
rarch = Oceananigans.Architectures.ReactantState()
varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi, e=1e-6)
GordonBell25.sync_states!(rmodel, vmodel)

@info "At the beginning:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@jit Oceananigans.initialize!(rmodel)
Oceananigans.initialize!(vmodel)

@jit Oceananigans.TimeSteppers.update_state!(rmodel)
Oceananigans.TimeSteppers.update_state!(vmodel)

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)

# rfake! = @compile sync=true raise=true fake_time_step_catke_equation!(rmodel)
# @showtime rfake!(rmodel)
# @showtime fake_time_step_catke_equation!(vmodel)

rfirst! = @compile sync=true raise=true GordonBell25.first_time_step!(rmodel)
@showtime rfirst!(rmodel)
@showtime GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
