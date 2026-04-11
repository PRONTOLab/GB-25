# julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_phase5_loop.jl --float-type=Float64 --target-float-type=Float32 --dimension=first

include("correctness_sharded_baroclinic_common.jl")

ctx = setup_context()
compare!(ctx, "At the beginning:")

initialize_and_update_state!(ctx)
compare!(ctx, "After initialization and update state:")

run_first_time_step!(ctx)
compare!(ctx, "After first time step:")

Nt = parse(Int, get(ENV, "GB_CORRECTNESS_LOOP_STEPS", "100"))
rNt = ConcreteRNumber(Nt)
rloop! = timed_phase("Compile loop!(rmodel, rNt)") do
    @compile compile_options=ctx.compile_options GordonBell25.loop!(ctx.rmodel, rNt)
end

@showtime rloop!(ctx.rmodel, rNt)
@showtime GordonBell25.loop!(ctx.vmodel, Nt)

compare!(ctx, "After a loop of $(Nt) steps:")
