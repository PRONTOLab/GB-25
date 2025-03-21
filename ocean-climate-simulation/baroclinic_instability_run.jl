using GordonBell25: baroclinic_instability_model_init
using Oceananigans.Architectures: ReactantState
using Oceananigans.Units
using Reactant

# Reactant.Compiler.SROA_ATTRIBUTOR[] = false

include("common.jl")
Ninner = ConcreteRNumber(3)

@info "Generating model..."
r_model = baroclinic_instability_model_init(ReactantState(), Δt=2minutes, resolution=1/4)
c_model = baroclinic_instability_model_init(CPU(), Δt=2minutes, resolution=1/4)

#=
c_u = c_model.velocities.u
c_v = c_model.velocities.v
c_b = c_model.tracers.b

r_u = r_model.velocities.u
r_v = r_model.velocities.v
r_b = r_model.tracers.b

copyto!(parent(r_u), Reactant.to_rarray(parent(c_u)))
copyto!(parent(r_v), Reactant.to_rarray(parent(c_v)))
copyto!(parent(r_b), Reactant.to_rarray(parent(c_b)))
=#

GC.gc(true); GC.gc(false); GC.gc(true)

@info "Compiling..."
r_update_state! = @compile sync=true raise=true Oceananigans.TimeSteppers.update_state!(r_model)
r_first_time_step! = @compile sync=true raise=true first_time_step!(r_model)
r_loop! = @compile sync=true raise=true loop!(r_model, Ninner)

r_update_state!(r_model)

first_time_step!(c_model)
r_first_time_step!(r_model)

@time loop!(c_model, 10)
@time r_loop!(r_model, ConcreteRNumber(10))

Nt = 100
@time loop!(c_model, Nt)
@time r_loop!(r_model, ConcreteRNumber(Nt))

rb = r_model.tracers.b
cb = c_model.tracers.b

rη = r_model.free_surface.η
cη = c_model.free_surface.η

ru = r_model.velocities.u
cu = c_model.velocities.u

rv = r_model.velocities.v
cv = c_model.velocities.v

rw = r_model.velocities.w
cw = c_model.velocities.w

rp = r_model.pressure.pHY′
cp = c_model.pressure.pHY′

using GLMakie

fig = Figure()

axc = Axis(fig[1, 1], title="Regular Oceananigans")
axr = Axis(fig[1, 2], title="With Reactant")
axd = Axis(fig[1, 3], title="Difference")

axuc = Axis(fig[2, 1], title="Regular Oceananigans")
axur = Axis(fig[2, 2], title="With Reactant")
axud = Axis(fig[2, 3], title="Difference")

axvc = Axis(fig[3, 1], title="Regular Oceananigans")
axvr = Axis(fig[3, 2], title="With Reactant")
axvd = Axis(fig[3, 3], title="Difference")

axwc = Axis(fig[4, 1], title="Regular Oceananigans")
axwr = Axis(fig[4, 2], title="With Reactant")
axwd = Axis(fig[4, 3], title="Difference")

axηc = Axis(fig[5, 1], title="Regular Oceananigans")
axηr = Axis(fig[5, 2], title="With Reactant")
axηd = Axis(fig[5, 3], title="Difference")

axpc = Axis(fig[6, 1], title="Regular Oceananigans")
axpr = Axis(fig[6, 2], title="With Reactant")
axpd = Axis(fig[6, 3], title="Difference")

# heatmap!(axc, view(cb, :, :, 1))
# heatmap!(axr, view(rb, :, :, 1))

cbp = parent(cb)
rbp = Array(parent(rb))
Δbp = cbp .- rbp

cup = parent(cu)
rup = Array(parent(ru))
Δup = cup .- rup

cvp = parent(cv)
rvp = Array(parent(rv))
Δvp = cvp .- rvp

cwp = parent(cw)
rwp = Array(parent(rw))
Δwp = cwp .- rwp

cpp = parent(cp)
rpp = Array(parent(rp))
Δpp = cpp .- rpp

Nz = size(parent(cbp), 3)
slider = Slider(fig[7, 1:3], range=1:Nz, startvalue=30)
k = slider.value

title = @lift string("Vertical level: ", $k)
Label(fig[0, 1:3], title)
axc = Axis(fig[1, 1], title="Regular Oceananigans")

cbk = @lift view(cbp, :, :, $k)
rbk = @lift view(rbp, :, :, $k)
Δbk = @lift view(Δbp, :, :, $k)

cuk = @lift view(cup, :, :, $k)
ruk = @lift view(rup, :, :, $k)
Δuk = @lift view(Δup, :, :, $k)

cvk = @lift view(cvp, :, :, $k)
rvk = @lift view(rvp, :, :, $k)
Δvk = @lift view(Δvp, :, :, $k)

cwk = @lift view(cwp, :, :, $k)
rwk = @lift view(rwp, :, :, $k)
Δwk = @lift view(Δwp, :, :, $k)

cpk = @lift view(cpp, :, :, $k)
rpk = @lift view(rpp, :, :, $k)
Δpk = @lift view(Δpp, :, :, $k)

heatmap!(axc, cbk, colormap=:magma, colorrange=extrema(cbp))
heatmap!(axr, rbk, colormap=:magma, colorrange=extrema(cbp))
heatmap!(axd, Δbk, colormap=:magma, colorrange=extrema(cbp))

heatmap!(axuc, cuk, colormap=:balance, colorrange=extrema(cup))
heatmap!(axur, ruk, colormap=:balance, colorrange=extrema(cup))
heatmap!(axud, Δuk, colormap=:balance, colorrange=extrema(cup))

heatmap!(axvc, cvk, colormap=:balance, colorrange=extrema(cvp))
heatmap!(axvr, rvk, colormap=:balance, colorrange=extrema(cvp))
heatmap!(axvd, Δvk, colormap=:balance, colorrange=extrema(cvp))

heatmap!(axwc, cwk, colormap=:balance, colorrange=extrema(cwp))
heatmap!(axwr, rwk, colormap=:balance, colorrange=extrema(cwp))
heatmap!(axwd, Δwk, colormap=:balance, colorrange=extrema(cwp))

cηp = parent(cη)
rηp = Array(parent(rη))
Δηp = cηp .- rηp

heatmap!(axηc, cηp[:, :, 1], colormap=:balance, colorrange=(-0.004, 0.004))
heatmap!(axηr, rηp[:, :, 1], colormap=:balance, colorrange=(-0.004, 0.004))
heatmap!(axηd, Δηp[:, :, 1], colormap=:balance, colorrange=(-0.004, 0.004))

heatmap!(axpc, cpk, colormap=:balance, colorrange=extrema(cpp))
heatmap!(axpr, rpk, colormap=:balance, colorrange=extrema(cpp))
heatmap!(axpd, Δpk, colormap=:balance, colorrange=extrema(cpp))

display(current_figure())

#=
for n = 1:100
    @time r_loop!(r_model, ConcreteRNumber(100))
    heatmap(view(rb, :, :, 1))
    display(current_figure())
    sleep(0.1)
end
=#

#=
Nt = 100
r_Nt = ConcreteRNumber(Nt)
@info "Running $_Ninner time-steps..."
r_loop!(model, r_Nt)
loop!(model, Nt)
=#

