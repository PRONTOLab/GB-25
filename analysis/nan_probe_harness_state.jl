include(joinpath(dirname(@__FILE__), "persistent_correctness_harness.jl"))

pair = build_model_pair(dbg)

println("NaN probe for freshly built/synced model pair")
println("grid = ", size(parent(Oceananigans.fields(pair.vmodel)[:u])))

for name in keys(Oceananigans.fields(pair.vmodel))
    rfield = Array(parent(Oceananigans.fields(pair.rmodel)[name]))
    vfield = Array(parent(Oceananigans.fields(pair.vmodel)[name]))
    nr = count(isnan, rfield)
    nv = count(isnan, vfield)
    if nr > 0 || nv > 0
        println("field=", name, " nan_r=", nr, " nan_v=", nv)
    end
end

# Mirror the exact compare call used in check_stage! initial comparison.
GordonBell25.compare_states(
    pair.rmodel,
    pair.vmodel;
    include_halos = true,
    throw_error = false,
    rtol = dbg.rtol,
    atol = dbg.atol,
)
