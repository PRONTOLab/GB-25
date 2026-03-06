module GordonBell25

export time_step!, loop!, try_code_hlo

include("model_utils.jl")
include("timestepping_utils.jl")
include("data_free_ocean_climate_model.jl")
include("baroclinic_instability_model.jl")
include("sharding_utils.jl")
include("correctness.jl")
include("precompile.jl")

end # module
