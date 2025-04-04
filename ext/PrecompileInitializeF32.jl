module PrecompileInitializeF32

using GordonBell25
using Reactant
using Oceananigans
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    FT = Float32
    Oceananigans.defaults.FloatType = FT
    Nx, Ny, Nz = 64, 32, 4
    arch = Oceananigans.Architectures.ReactantState()

    if Reactant.XLA.REACTANT_XLA_RUNTIME == "PJRT"
        client = Reactant.XLA.PJRT.CPUClient(; checkcount=false)
    elseif Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"
        client = Reactant.XLA.IFRT.CPUClient(; checkcount=false)
    else
        error("Unsupported runtime: $(Reactant.XLA.REACTANT_XLA_RUNTIME)")
    end

    @compile_workload begin
        model = GordonBell25.baroclinic_instability_model(arch; resolution=4, Δt=60, Nz=4, grid_type=:simple_lat_lon)
        Reactant.compile(Oceananigans.initialize!, (model,); client, optimize=:all)
    end

    XLA.free_client(client)
    client.client = C_NULL

    # Reset float type
    Oceananigans.defaults.FloatType = Float64
    Reactant.clear_oc_cache()
end

end # module

