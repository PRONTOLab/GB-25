using Oceananigans
using Reactant

function simple_model(arch)
    Nx = Nz = 16
    grid = RectilinearGrid(arch, size=(Nx, 1, Nz), x=(-Nx/2, Nx/2), y=(0, 1), z=(0, Nz))
    free_surface = ExplicitFreeSurface()
    model = HydrostaticFreeSurfaceModel(; grid, free_surface, buoyancy=BuoyancyTracer(), tracers=:b)
    bᵢ(x, y, z) = exp(-x^2)
    set!(model, b=bᵢ)
    return model
end

c_model = simple_model(CPU())
r_model = simple_model(Oceananigans.Architectures.ReactantState())

using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!

c_p = c_model.pressure.pHY′
r_p = r_model.pressure.pHY′
@time update_hydrostatic_pressure!(c_p, CPU(), c_model.grid, c_model.buoyancy, c_model.tracers)

r_update_hydrostatic_pressure = @compile sync=true, raise=true update_hydrostatic_pressure!(r_p, ReactantState(), r_model.grid, r_model.buoyancy, r_model.tracers)
@time r_update_hydrostatic_pressure!(r_p, ReactantState(), r_model.grid, r_model.buoyancy, r_model.tracers)

pc = c_model.pressure.pHY′
pr = r_model.pressure.pHY′

pcp = interior(pc, :, :, 1)
pcr = Array(interior(pr, :, :, 1))

@show mean(pcp .- pcr)
