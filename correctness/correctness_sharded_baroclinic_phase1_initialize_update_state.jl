# julia --project=. -O0 --threads=16 correctness/correctness_sharded_baroclinic_phase1_initialize_update_state.jl --float-type=Float64 --target-float-type=Float32 --dimension=first

include("correctness_sharded_baroclinic_common.jl")

ctx = setup_context()
compare!(ctx, "At the beginning:")

initialize_and_update_state!(ctx)
compare!(ctx, "After initialization and update state:")
