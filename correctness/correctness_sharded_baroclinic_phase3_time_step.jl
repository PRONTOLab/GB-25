# julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_phase3_time_step.jl --float-type=Float64 --target-float-type=Float32 --dimension=first

include("correctness_sharded_baroclinic_common.jl")

ctx = setup_context()
compare!(ctx, "At the beginning:")

initialize_and_update_state!(ctx)
compare!(ctx, "After initialization and update state:")

run_first_time_step!(ctx)
compare!(ctx, "After first time step:")

rstep! = compile_time_step!(ctx)

@info "Warm up rmodel:"
@showtime rstep!(ctx.rmodel)
@showtime rstep!(ctx.rmodel)

@info "Warm up vmodel:"
@showtime GordonBell25.time_step!(ctx.vmodel)
@showtime GordonBell25.time_step!(ctx.vmodel)

Nt = parse(Int, get(ENV, "GB_CORRECTNESS_TIMESTEP_STEPS", "10"))

@info "Time step with Reactant ($(Nt) steps):"
for _ in 1:Nt
    @showtime rstep!(ctx.rmodel)
end

@info "Time step vanilla ($(Nt) steps):"
for _ in 1:Nt
    @showtime GordonBell25.time_step!(ctx.vmodel)
end

compare!(ctx, "After $(Nt) time_step! calls:")
