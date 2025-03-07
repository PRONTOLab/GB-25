using KernelAbstractions, Reactant, Oceananigans

@kernel kernel!() = nothing
function launch_kernel()
    dev = KernelAbstractions.get_backend(Reactant.to_rarray(Float32[]))
    workgroup, worksize = (16, 16), (180, 85)
    kernel!(dev, workgroup, worksize)()
end
