using Oceananigans
using Reactant
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf
using JSON

Reactant.Compiler.Raise[] = true

function loop!(model, Δt, Ninner)
    @trace for _ = 1:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end

params = open(JSON.parse, joinpath(@__DIR__, "params.json"))
in_precis = eval(Meta.parse(params["precis"]))
in_arch = eval(Meta.parse(params["arch"]))
in_resol = eval(Meta.parse(params["resol"]))

# Set default floating point type
FT = in_precis
Oceananigans.defaults.FloatType = FT

# Architecture
arch = in_arch
resolution = in_resol
Lz = 1kilometers     # depth [m]
Ny = Base.Int(20 / resolution)
Nz = 50
N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
Δb = 0.005 # [m/s²] buoyancy difference
φ₀ = 50
closure = VerticalScalarDiffusivity(FT; κ=1e-5, ν=1e-4)
prefix = joinpath(@__DIR__, "baroclinic_adjustment_$FT")
stop_time = 800days

@info "Nx, Ny, Nz = $Ny, $Ny, $Nz"

grid = LatitudeLongitudeGrid(arch,
                             topology = (Periodic, Bounded, Bounded),
                             size = (Ny, Ny, Nz),
                             longitude = (-10, 10),
                             latitude = (φ₀ - 10, φ₀ + 10),
                             z = (-Lz, 0),
                             halo = (6, 6, 6))

model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENOVectorInvariant(order=5),
                                    tracer_advection = WENO(order=5))

# Parameters
parameters = (; N², Δb, φ₀, Δφ = 20)

@inline function bᵢ(λ, φ, z, p)
    γ = π/2 - 2π * (p.φ₀ - φ) / p.Δφ
    b = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return p.N² * z + p.Δb * b
end

ϵb = 1e-2 * Δb # noise amplitude
Random.seed!(1234)
bᵢ(x, y, z) = bᵢ(x, y, z, parameters) + ϵb * randn()
set!(model, b=bᵢ)

# time step
dx = minimum_xspacing(grid)
Δt = 0.15 * dx / 2 # c * dx / max(U)

# u, v, w = model.velocities
# e = @at (Center, Center, Center) (u^2 + v^2) / 2
# E = Average(e, dims=(1, 2, 3))
# ke_ow = JLD2OutputWriter(model, (; E),
#                          filename = prefix * "_kinetic_energy.jld2",
#                          schedule = TimeInterval(1days),
#                          overwrite_existing = true)

# output_writers[:ke] = ke_ow

# Nz = size(grid, 3)
# b = model.tracers.b
# ζ = ∂x(v) - ∂y(u)
# fields = (; u, v, w, b, ζ)
# f_ow = JLD2OutputWriter(model, fields,
#                         filename = prefix * "_fields.jld2",
#                         indices = (:, :, Nz),
#                         schedule = TimeInterval(10days),
#                         overwrite_existing = true)

# output_writers[:fields] = f_ow

@info "Compiling..."
Ninner = ConcreteRNumber(10) # this must be traced, so that we can change it later.
r_loop! = @compile loop!(model, Δt, Ninner)
r_first_time_step! = @compile Oceananigans.TimeSteppers.first_time_step!(model, Δt)

# Do one initial loop
@info "Running initial loop..."
Nprint = 10
r_first_time_step!(model, Δt)
r_loop!(model, Δt, ConcreteRNumber(Nprint-1))

# Now set an outer loop in motion
Ntotal = ceil(Int, stop_time / Δt) # 1000 # or, compute this with a stop_time + Δt
Nouter = ceil(Int, (Ntotal - 1) / Nprint)

wall_clock = Ref(time_ns())

@info "Running..."
for outer = 1:Nouter

    r_loop!(model, Δt, ConcreteRNumber(Nprint))

    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("Iter: %d, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s",
                   Nouter * Nprint, prettytime(elapsed),
                   maximum(abs, model.velocities.u),
                   maximum(abs, model.velocities.v),
                   maximum(abs, model.velocities.w))

    @info msg

    wall_clock[] = time_ns()

    # Oceananigans.OutputWriters.write_output!(output_writer, model)
end

@info "Done!"
