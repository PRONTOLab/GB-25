ENV["JULIA_DEBUG"] = "Reactant_jll,Reactant"
using GordonBell25
using Oceananigans
const parsed_args = GordonBell25.parse_baroclinic_instability_args(;
    grid_x_default = 128,
    grid_y_default = 128,
    grid_z_default = 16,
)

default_float_type = GordonBell25.float_type_from_args(parsed_args)
Oceananigans.defaults.FloatType = default_float_type
using Reactant
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true

using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.BoundaryConditions: update_boundary_condition!, replace_horizontal_vector_halos!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models: update_model_field_time_series!
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.Fields: tupled_fill_halo_regions!

import Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!
import Oceananigans.TimeSteppers: update_state!

# Mask immersed fields
function mask_immersed_model_fields!(model, grid)
    η = Oceananigans.Models.HydrostaticFreeSurfaceModels.displacement(model.free_surface)
    fields_to_mask = merge(model.auxiliary_fields, prognostic_fields(model))

    foreach(fields_to_mask) do field
        if field !== η
            mask_immersed_field!(field)
        end
    end
    mask_immersed_field_xy!(η, k=size(grid, 3)+1)

    return nothing
end

throw_error = true
include_halos = true
rtol = sqrt(eps(default_float_type))
atol = 0

model_kw = (
    halo = (8, 8, 8),
    Δt = 1e-9,
)

H = 8
Rx = Ry = 1
Tx = parsed_args["grid-x"] * Rx
Ty = parsed_args["grid-y"] * Ry
Nz = parsed_args["grid-z"]

Nx = Tx - 2H
Ny = Ty - 2H

rarch = Oceananigans.Architectures.ReactantState()
varch = CPU()
rmodel = GordonBell25.baroclinic_instability_model(rarch, Nx, Ny, Nz; model_kw...)
vmodel = GordonBell25.baroclinic_instability_model(varch, Nx, Ny, Nz; model_kw...)
@show vmodel
@show rmodel

ui = 1e-3 .* rand(size(vmodel.velocities.u)...)
vi = 1e-3 .* rand(size(vmodel.velocities.v)...)
set!(vmodel, u=ui, v=vi)
GordonBell25.sync_states!(rmodel, vmodel)

@info "At the beginning:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

compile_options = CompileOptions(; sync=true, raise=true, strip_llvm_debuginfo=true, strip=["enzymexla.kernel_call", "(::Reactant.Compiler.LLVMFunc", "ka_with_reactant", "(::KernelAbstractions.Kernel", "var\"#_launch!;_launch!"], multifloat=GordonBell25.multifloat_from_args(parsed_args))

@jit compile_options=compile_options Oceananigans.initialize!(rmodel)
Oceananigans.initialize!(vmodel)

    grid = rmodel.grid
    callbacks = []
    compute_tendencies = true

    function step1_mask!(model)
        @apply_regionally mask_immersed_model_fields!(model, model.grid)
    end
    function step2_time_series!(model)
        @apply_regionally update_model_field_time_series!(model, model.clock)
    end
    function step3_bc!(model)
        @apply_regionally update_boundary_condition!(fields(model), model)
    end
    function step4_halo!(model)
        tupled_fill_halo_regions!(prognostic_fields(model), model.grid, model.clock, fields(model), async=true)
    end
    function step5_vector_halo!(model)
        @apply_regionally replace_horizontal_vector_halos!(model.velocities, model.grid)
    end
    function step6_aux!(model)
        @apply_regionally compute_auxiliaries!(model)
    end

    r_step1! = @compile compile_options=compile_options step1_mask!(rmodel)
    r_step1!(rmodel)
    step1_mask!(vmodel)
    @info "After mask_immersed_model_fields!:"
    GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

    r_step2! = @compile compile_options=compile_options step2_time_series!(rmodel)
    r_step2!(rmodel)
    step2_time_series!(vmodel)
    @info "After update_model_field_time_series!:"
    GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

    r_step3! = @compile compile_options=compile_options step3_bc!(rmodel)
    r_step3!(rmodel)
    step3_bc!(vmodel)
    @info "After update_boundary_condition!:"
    GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

    r_step4! = @compile compile_options=compile_options step4_halo!(rmodel)
    r_step4!(rmodel)
    step4_halo!(vmodel)
    @info "After tupled_fill_halo_regions!:"
    GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

    r_step5! = @compile compile_options=compile_options step5_vector_halo!(rmodel)
    r_step5!(rmodel)
    step5_vector_halo!(vmodel)
    @info "After replace_horizontal_vector_halos!:"
    GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

    r_step6! = @compile compile_options=compile_options step6_aux!(rmodel)
    r_step6!(rmodel)
    step6_aux!(vmodel)
    @info "After compute_auxiliaries!:"
    GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

@info "After initialization and update state:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)
rfirst! = @compile compile_options=compile_options GordonBell25.first_time_step!(rmodel)
@showtime rfirst!(rmodel)
@showtime GordonBell25.first_time_step!(vmodel)

@info "After first time step:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

rstep! = @compile compile_options=compile_options GordonBell25.time_step!(rmodel)

@info "Warm up:"
@showtime rstep!(rmodel)
@showtime rstep!(rmodel)
@showtime GordonBell25.time_step!(vmodel)
@showtime GordonBell25.time_step!(vmodel)

Nt = 10
@info "Time step with Reactant:"
for _ in 1:Nt
    @showtime rstep!(rmodel)
end

@info "Time step vanilla:"
for _ in 1:Nt
    @showtime GordonBell25.time_step!(vmodel)
end

@info "After $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

GordonBell25.sync_states!(rmodel, vmodel)
@jit compile_options=compile_options Oceananigans.TimeSteppers.update_state!(rmodel)

@info "After syncing and updating state again:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile compile_options=compile_options GordonBell25.loop!(rmodel, rNt)
@showtime rloop!(rmodel, rNt)
@showtime GordonBell25.loop!(vmodel, Nt)

@info "After a loop of $(Nt) steps:"
GordonBell25.compare_states(rmodel, vmodel; include_halos, throw_error, rtol, atol)
