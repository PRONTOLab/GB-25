# julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_phase4_resync_update_state.jl --float-type=Float64 --target-float-type=Float32 --dimension=first

include("correctness_sharded_baroclinic_common.jl")

ctx = setup_context()
compare!(ctx, "At the beginning:")

initialize_and_update_state!(ctx)
compare!(ctx, "After initialization and update state:")

run_first_time_step!(ctx)
compare!(ctx, "After first time step:")

rstep! = compile_time_step!(ctx)
@showtime rstep!(ctx.rmodel)
@showtime GordonBell25.time_step!(ctx.vmodel)
compare!(ctx, "After one time_step!:")

resync_and_update_state!(ctx)
compare!(ctx, "After syncing and updating state again:")
