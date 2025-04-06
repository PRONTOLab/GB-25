using GordonBell25: first_time_step!, loop!, try_code_hlo, preamble
using GordonBell25: data_free_ocean_climate_model_init, PROFILE
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState

PROFILE[] = true

preamble()

@info "Generating model..."
model = data_free_ocean_climate_model_init(ReactantState(), resolution=8)

GC.gc(true); GC.gc(false); GC.gc(true)

failed = false
Ninner = ConcreteRNumber(2)

# Pre-raise IR
@info "Compiling before raise kernels..."
before_raise_first = try_code_hlo() do
    @code_hlo optimize=:before_raise raise=true first_time_step!(model)
end

before_raise_loop = try_code_hlo() do
    @code_hlo optimize=:before_raise raise=true loop!(model, Ninner)
end

# Unoptimized HLO
@info "Compiling unoptimised kernel..."
unopt_first = try_code_hlo() do
    @code_hlo optimize=false raise=true first_time_step!(model)
end

unopt_loop = try_code_hlo() do
    @code_hlo optimize=false raise=true loop!(model, Ninner)
end

# Optimized HLO
@info "Compiling optimised kernel..."
opt_first = try_code_hlo() do
    @code_hlo optimize=:before_jit raise=true first_time_step!(model)
end

opt_loop = try_code_hlo() do
    @code_hlo optimize=:before_jit raise=true loop!(model, Ninner)
end

for type in ("before_raise", "unopt", "opt"), name in ("first", "loop"), debug in (true, false)
    open("$(type)_ocean_climate_simulation_$(name)$(debug ? "_debug" : "").mlir", "w") do io
        show(IOContext(io, :debug => debug), getfield(Main, @eval Symbol($type, "_", $name)))
    end
end

if failed
    error("compilation failed")
end
