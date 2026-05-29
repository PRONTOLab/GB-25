module GordonBell25

using JLD2

export first_time_step!, time_step!, loop!, try_code_hlo
export parse_baroclinic_instability_args, float_type_from_args, multifloat_from_args
export local_shards_to_host, save_sharded_fields, save_model_state, resolve_z_indices
export load_global_field, load_checkpoint_metadata, load_all_fields, extract_model_fields

include("arg_parsing.jl")
include("model_utils.jl")
include("timestepping_utils.jl")
include("data_free_ocean_climate_model.jl")
include("baroclinic_instability_model.jl")
include("sharding_utils.jl")
include("sharded_io.jl")
include("correctness.jl")
include("precompile.jl")
include("interpolate.jl")

end # module
