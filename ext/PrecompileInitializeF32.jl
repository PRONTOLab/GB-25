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

    @compile_workload begin
        model = GordonBell25.baroclinic_instability_model(arch; resolution=4, Δt=60, Nz=4, grid=:simple_lat_lon)
        compiled! = @compile Oceananigans.initialize!(model)
    end

    # Reset float type
    Oceananigans.defaults.FloatType = Float64
    Reactant.clear_oc_cache()
end

end # module

