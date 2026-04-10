using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: moist_baroclinic_wave_model
using Breeze.AtmosphereModels: dynamics_density
using Oceananigans
using Oceananigans.Architectures: GPU
using CUDA
using JLD2
using Dates

Oceananigans.defaults.FloatType = Float32

preamble()

H = 8

@info "Generating atmosphere model on plain CUDA GPU (no Reactant)..." now(UTC)
arch = GPU()
model = moist_baroclinic_wave_model(arch; Nλ=1440, Nφ=680, Nz=64, Δt=30.0, halo=(H, H, H))
@show model

@info "First time step..." now(UTC)
@time "first time step" first_time_step!(model)

@info "Stepping..." now(UTC)
Nt = 100
@time "loop $Nt steps" for _ in 1:Nt
    time_step!(model)
end

@info "Done stepping" now(UTC) model.clock

# Save the prognostic dynamics + thermo + water vapor state (no microphysics
# tracers — cloud liquid/ice/rain/snow are skipped). Stored as the conservative
# variables that AtmosphereModel actually time-steps, so they can be reloaded
# directly as initial conditions on the same grid.
checkpoint_dir  = joinpath(@__DIR__, "checkpoints")
mkpath(checkpoint_dir)
jobid           = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH-MM-SS")
checkpoint_path = joinpath(checkpoint_dir, "atmosphere_spinup_state_quarter_degree_$(jobid).jld2")

@info "Saving atmosphere full state (ρ, ρu, ρv, ρw, ρθ, ρqᵛ)" checkpoint_path
@time "checkpoint save" begin
    # Field-level accessors; `interior` strips halos.
    ρ_field   = dynamics_density(model.dynamics)
    ρu_field  = model.momentum.ρu
    ρv_field  = model.momentum.ρv
    ρw_field  = model.momentum.ρw
    ρθ_field  = model.formulation.potential_temperature_density
    ρqv_field = model.moisture_density
    JLD2.jldsave(checkpoint_path;
                 ρ   = Array(interior(ρ_field)),
                 ρu  = Array(interior(ρu_field)),
                 ρv  = Array(interior(ρv_field)),
                 ρw  = Array(interior(ρw_field)),
                 ρθ  = Array(interior(ρθ_field)),
                 ρqᵛ = Array(interior(ρqv_field)),
                 Nλ  = size(model.grid, 1),
                 Nφ  = size(model.grid, 2),
                 Nz  = size(model.grid, 3),
                 time      = model.clock.time,
                 iteration = model.clock.iteration,
                 last_Δt   = model.clock.last_Δt)
end
@info "Saved" checkpoint_path filesize(checkpoint_path)

@info "Done!" now(UTC)
