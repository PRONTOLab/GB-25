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
        "--target-float-type"
            help = "the float type for execution (Float64/f64, Float32/f32, Float16/f16, BFloat16/bf16, Float8E5M2/f8E5M2/f8, Float8E4M3/f8E4M3) or the empty string for no lowering"
            default = ""
            arg_type = String
        "--limbs"
            help = "Number of lower-precision limbs in the multifloat lowering"
            default = 2
            arg_type = Int
        "--dimension"
	    help = "Multifloat expansion dimension (first, last, tuple)"
            default = "first"
            arg_type = String
    end
    return parse_args(ARGS, s)
end

struct Float8E5M2
end

struct Float8E4M3
end

function float_type_from_string(s::String)
    if s ∈ ("Float64", "f64")
        return Float64
    elseif s ∈ ("Float32", "f32")
        return Float32
    elseif s ∈ ("Float16", "f16")
        return Float16
    elseif s ∈ ("BFloat16", "bf16")
        return Core.BFloat16
    elseif s in ("Float8E5M2", "f8E5M2", "f8")
	return Float8E5M2
    elseif s in ("Float8E4M3", "f8E4M3")
	return Float8E4M3
    else
        throw(AssertionError("Unknown float type $s"))
    end
end

"""
    float_type_from_args(parsed_args)

Convert the "--float-type" string from `parse_baroclinic_instability_args` to a Julia type.
"""
function float_type_from_args(parsed_args)
    float_type_from_string(parsed_args["float-type"])
end

function float_type_to_string(s)
   if s == Float64
	return "f64"
   elseif s == Float32
	return "f32"
   elseif s == Float16
	return "f16"
   elseif s == Core.BFloat16
	return "bf16"
   elseif s == Float8E5M2
	return "f8E5M2"
   elseif s == Float8E4M3
	return "f8E4M3"
   else
	throw(AssertionError("Unknown float type $s"))
   end
end

function multifloat_from_args(parsed_args)
    if parsed_args["target-float-type"] == ""
	return nothing
    end
    # normalize the string
    source = float_type_to_string(float_type_from_string(parsed_args["float-type"]))
    target = float_type_to_string(float_type_from_string(parsed_args["float-type-target"]))
    return Reactant.Compiler.MultiFloatOptions(source, target, parsed_args["dimension"], parsed_args["limbs"])
end

