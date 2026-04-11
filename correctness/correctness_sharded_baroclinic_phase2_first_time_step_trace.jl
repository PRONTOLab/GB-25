# julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_phase2_first_time_step_trace.jl --grid-x=64 --grid-y=64 --grid-z=16 --float-type=Float64 --target-float-type=Float32 --dimension=first
#
# For local single-process 4-TPU debugging:
#   TPU_VISIBLE_DEVICES=0,1,2,3 GB_SKIP_DISTRIBUTED_INIT=1 julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_phase2_first_time_step_trace.jl ...

include("correctness_sharded_baroclinic_common.jl")

function _aligned_cpu_view(rfield, vfield)
    r = Array(parent(rfield))
    v = Array(parent(vfield))
    Nx, Ny, Nz = size(r)
    return r, view(v, 1:Nx, 1:Ny, 1:Nz)
end

function _focus_compare_one(name, rfield, vfield, rtol, atol)
    r, v = _aligned_cpu_view(rfield, vfield)
    δ = r .- v
    maxδ, idx = findmax(abs, δ)
    approx_equal = isapprox(r, v; rtol, atol)
    @info "focus compare" field=name approx_equal max_abs_delta=maxδ max_r=maximum(abs, r) max_v=maximum(abs, v) idx=Tuple(idx)
    return approx_equal
end

function focus_compare!(ctx, label)
    @info label
    all_ok = true
    all_ok &= _focus_compare_one("u", ctx.rmodel.velocities.u, ctx.vmodel.velocities.u, ctx.rtol, ctx.atol)
    all_ok &= _focus_compare_one("v", ctx.rmodel.velocities.v, ctx.vmodel.velocities.v, ctx.rtol, ctx.atol)
    all_ok &= _focus_compare_one("w", ctx.rmodel.velocities.w, ctx.vmodel.velocities.w, ctx.rtol, ctx.atol)
    all_ok &= _focus_compare_one("Gⁿ.u", ctx.rmodel.timestepper.Gⁿ[:u], ctx.vmodel.timestepper.Gⁿ[:u], ctx.rtol, ctx.atol)
    all_ok &= _focus_compare_one("Gⁿ.v", ctx.rmodel.timestepper.Gⁿ[:v], ctx.vmodel.timestepper.Gⁿ[:v], ctx.rtol, ctx.atol)
    all_ok &= _focus_compare_one("G⁻.u", ctx.rmodel.timestepper.G⁻[:u], ctx.vmodel.timestepper.G⁻[:u], ctx.rtol, ctx.atol)
    all_ok &= _focus_compare_one("G⁻.v", ctx.rmodel.timestepper.G⁻[:v], ctx.vmodel.timestepper.G⁻[:v], ctx.rtol, ctx.atol)
    return all_ok
end

function maybe_full_compare!(ctx, label)
    do_full_compare = get(ENV, "GB_TRACE_FULL_COMPARE", "0") == "1"
    do_full_compare || return
    compare!(ctx, label)
end

function manual_first_time_step_trace!(ctx)
    # Keep both models at exactly the same pre-step state.
    GordonBell25.sync_states!(ctx.rmodel, ctx.vmodel)

    focus_compare!(ctx, "Trace start (pre time_step)")
    maybe_full_compare!(ctx, "Trace start (pre time_step, full)")

    Δt_r = ctx.rmodel.clock.last_Δt
    Δt_v = ctx.vmodel.clock.last_Δt
    Δt_r == Δt_v || error("Clock mismatch before trace: Δt_r=$Δt_r, Δt_v=$Δt_v")
    Δt = Δt_r

    # Stage A: initial update_state! inside time_step! when iteration == 0.
    timed_phase("Trace A: update_state!(rmodel; compute_tendencies=true)") do
        @jit compile_options=ctx.compile_options Oceananigans.TimeSteppers.update_state!(ctx.rmodel, []; compute_tendencies=true)
    end
    Oceananigans.TimeSteppers.update_state!(ctx.vmodel, []; compute_tendencies=true)
    focus_compare!(ctx, "After Trace A")
    maybe_full_compare!(ctx, "After Trace A (full)")

    # Stage B: AB2/Euler state advance.
    χ₀r = ctx.rmodel.timestepper.χ
    χ₀v = ctx.vmodel.timestepper.χ
    minus_point_five_r = convert(eltype(ctx.rmodel.grid), -0.5)
    minus_point_five_v = convert(eltype(ctx.vmodel.grid), -0.5)
    ctx.rmodel.timestepper.χ = minus_point_five_r
    ctx.vmodel.timestepper.χ = minus_point_five_v

    timed_phase("Trace B: ab2_step!(rmodel, Δt)") do
        @jit compile_options=ctx.compile_options Oceananigans.TimeSteppers.ab2_step!(ctx.rmodel, Δt)
    end
    Oceananigans.TimeSteppers.ab2_step!(ctx.vmodel, Δt)
    focus_compare!(ctx, "After Trace B")
    maybe_full_compare!(ctx, "After Trace B (full)")

    Oceananigans.TimeSteppers.tick!(ctx.rmodel.clock, Δt)
    Oceananigans.TimeSteppers.tick!(ctx.vmodel.clock, Δt)
    ctx.rmodel.clock.last_Δt = Δt
    ctx.vmodel.clock.last_Δt = Δt
    ctx.rmodel.clock.last_stage_Δt = Δt
    ctx.vmodel.clock.last_stage_Δt = Δt

    # Stage C: pressure correction + cache previous tendencies.
    timed_phase("Trace C1: calculate_pressure_correction!(rmodel, Δt)") do
        @jit compile_options=ctx.compile_options Oceananigans.TimeSteppers.calculate_pressure_correction!(ctx.rmodel, Δt)
    end
    Oceananigans.TimeSteppers.calculate_pressure_correction!(ctx.vmodel, Δt)

    timed_phase("Trace C2: correct_velocities_and_cache_previous_tendencies!(rmodel, Δt)") do
        @jit compile_options=ctx.compile_options Oceananigans.TimeSteppers.correct_velocities_and_cache_previous_tendencies!(ctx.rmodel, Δt)
    end
    Oceananigans.TimeSteppers.correct_velocities_and_cache_previous_tendencies!(ctx.vmodel, Δt)
    focus_compare!(ctx, "After Trace C")
    maybe_full_compare!(ctx, "After Trace C (full)")

    # Stage D: final update_state!(...; compute_tendencies=true).
    timed_phase("Trace D: final update_state!(rmodel; compute_tendencies=true)") do
        @jit compile_options=ctx.compile_options Oceananigans.TimeSteppers.update_state!(ctx.rmodel, []; compute_tendencies=true)
    end
    Oceananigans.TimeSteppers.update_state!(ctx.vmodel, []; compute_tendencies=true)
    focus_compare!(ctx, "After Trace D")
    maybe_full_compare!(ctx, "After Trace D (full)")

    # Restore AB2 parameters.
    ctx.rmodel.timestepper.χ = χ₀r
    ctx.vmodel.timestepper.χ = χ₀v

    return nothing
end

ctx = setup_context()
compare!(ctx, "At the beginning:")

initialize_and_update_state!(ctx)
compare!(ctx, "After initialization and update state:")

manual_first_time_step_trace!(ctx)
