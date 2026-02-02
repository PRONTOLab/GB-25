using GordonBell25: first_time_step!, time_step!, loop!, preamble
using GordonBell25: data_free_ocean_climate_model_init
using Oceananigans.Architectures: ReactantState
using Reactant

using InteractiveUtils

using Oceananigans

using Oceananigans: initialize!
using Oceananigans.TimeSteppers: update_state!, time_step!

using ClimaOcean.OceanSeaIceModels: interpolate_state!

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

preamble()

Ninner = ConcreteRNumber(3)

@info "Generating model..."
coupled_model = data_free_ocean_climate_model_init(ReactantState(); resolution=4, Nz=10)
Ninner = ConcreteRNumber(2)

GC.gc(true); GC.gc(false); GC.gc(true)


@info "Compiling..."

atmosphere = coupled_model.atmosphere
exchanger  = coupled_model.interfaces.exchanger
grid       = exchanger.grid

@show @which interpolate_state!(exchanger.atmosphere, grid, atmosphere, coupled_model)

rfirst! = @compile raise=true sync=true interpolate_state!(exchanger.atmosphere, grid, atmosphere, coupled_model)

@info "Done!"
