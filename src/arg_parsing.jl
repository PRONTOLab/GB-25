using ArgParse, BFloat16s

"""
    parse_baroclinic_instability_args(; grid_x_default, grid_y_default, grid_z_default)

Parse standard CLI arguments for baroclinic instability simulations.
Returns a Dict with keys "grid-x", "grid-y", "grid-z", "float-type".
"""
function parse_baroclinic_instability_args(;
    grid_x_default::Int,
    grid_y_default::Int,
    grid_z_default::Int,
)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--grid-x"
            help = "Base factor for number of grid points on the x axis."
            default = grid_x_default
            arg_type = Int
        "--grid-y"
            help = "Base factor for number of grid points on the y axis."
            default = grid_y_default
            arg_type = Int
        "--grid-z"
            help = "Base factor for number of grid points on the z axis."
            default = grid_z_default
            arg_type = Int
        "--float-type"
            help = "The default Oceananigans float type (Float64/f64, Float32/f32, Float16/f16, BFloat16/bf16)"
            default = "Float64"
            arg_type = String
    end
    return parse_args(ARGS, s)
end

"""
    float_type_from_args(parsed_args)

Convert the "--float-type" string from `parse_baroclinic_instability_args` to a Julia type.
"""
function float_type_from_args(parsed_args)
    s = parsed_args["float-type"]
    if s ∈ ("Float64", "f64")
        return Float64
    elseif s ∈ ("Float32", "f32")
        return Float32
    elseif s ∈ ("Float16", "f16")
        return Float16
    elseif s ∈ ("BFloat16", "bf16")
        return Core.BFloat16
    else
        throw(AssertionError("Unknown float type $s"))
    end
end
