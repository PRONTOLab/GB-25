include("common.jl")

write(@show(joinpath(mktempdir(; cleanup=false), "module.mlir")), """
#loc = loc(unknown)
module @reactant_Base.Br... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<1xf32> {reactant.donated} loc(unknown)) -> tensor<1xf32> {
    %0 = stablehlo.sine %arg0 : tensor<1xf32> loc(#loc2)
    return %0 : tensor<1xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/giordano/.julia/dev/Reactant/src/Ops.jl":283:0)
#loc2 = loc("sine"(#loc1))
""")
