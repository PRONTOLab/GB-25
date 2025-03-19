module GordonBell25

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: Architectures
using Reactant

using ClimaOcean
using ClimaOcean.OceanSeaIceModels.InterfaceComputations: FixedIterations, ComponentInterfaces

using CFTime
using Dates
using Printf
using Profile
using Serialization

const PROFILE = Ref(false)

macro gbprofile(name::String, expr::Expr)
    return quote
        if $(PROFILE)[]
            $(Profile.clear)()
            $(Profile.init)(; delay=0.1)
            out = $(Profile).@profile $(esc(expr))
            open(string("profile_", $(esc(name)), ".txt"), "w") do s
                println(s, "# Showing profile of")
                println(s, "#     ", $(string(expr)))
                println(s, "# at ", $(string(__source__)))
                $(Profile.print)(IOContext(s, :displaysize => (48, 1000)))
            end
            $(Serialization.serialize)(string("profile_", $(esc(name)), ".dat"), $(Profile).retrieve())
            $(Profile.clear)()
            out
        else
            $(esc(expr))
        end
    end
end

include("data_free_ocean_model.jl")
include("ocean_climate_model.jl")
include("sharding_utils.jl")
include("precompile.jl")

end # module
