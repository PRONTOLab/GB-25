using GordonBell25
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant

function zero_halos(w)
    v = view(parent(w), :, :, 5:6)
    v .= 0
    v = view(parent(w), :, :, 1:2)
    v .= 0
    v = view(parent(w), :, 5:6, :)
    v .= 0
    v = view(parent(w), :, 1:2, :)
    v .= 0
    return nothing
end

function set_indexed!(w)
    N = size(parent(w))
    copyto!(parent(w), Reactant.to_rarray(reshape(collect(1:prod(N)), N...)))
    return w
end

kw = (size=(3, 2, 2), halo=(2, 2, 2), latitude = (-80, 80), longitude = (0, 360), z=(0, 1))
rgrid = LatitudeLongitudeGrid(ReactantState(); kw...)
vgrid = LatitudeLongitudeGrid(CPU(); kw...)
rw = ZFaceField(rgrid)
vw = ZFaceField(vgrid)

set_indexed!(rw)
@jit zero_halos(rw)
interior(vw) .= Array(interior(rw))

@info "Not traced"
@jit Oceananigans.BoundaryConditions.fill_halo_regions!(rw)
Oceananigans.BoundaryConditions.fill_halo_regions!(vw)
GordonBell25.compare_parent("w", rw, vw)

display(vw.data[0:4, 0:3, 2])
crw = Oceananigans.Architectures.on_architecture(CPU(), rw.data)
display(crw[0:4, 0:3, 2])

@info "Try again, traced"
set_indexed!(rw)
@jit zero_halos(rw)
interior(vw) .= Array(interior(rw))

function myloop!(w, Nt)
    @trace track_numbers=false for _ = 1:Nt
        Oceananigans.BoundaryConditions.fill_halo_regions!(w)
    end
end

Nt = 100
rNt = ConcreteRNumber(Nt)
rloop! = @compile sync=true raise=true myloop!(rw, rNt)
@time rloop!(rw, rNt)
@time myloop!(vw, Nt)

GordonBell25.compare_parent("w", rw, vw)

display(vw.data[0:4, 0:3, 2])
crw = Oceananigans.Architectures.on_architecture(CPU(), rw.data)
display(crw[0:4, 0:3, 1])
display(crw[0:4, 0:3, 2])
display(crw[0:4, 0:3, 3])

