using Reactant
Reactant.Ops.LARGE_CONSTANT_THRESHOLD[] = 100
using Oceananigans

# https://github.com/CliMA/Oceananigans.jl/blob/d9b3b142d8252e8e11382d1b3118ac2a092b38a2/src/Grids/orthogonal_spherical_shell_grid.jl#L14
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{Oceananigans.Grids.OrthogonalSphericalShellGrid{FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, Z, Map, CC, FC, CF, FF, Arch}
    FT2 = Reactant.traced_type_inner(FT, seen, mode, track_numbers, sharding, runtime)
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    Map2 = Reactant.traced_type_inner(Map, seen, mode, track_numbers, sharding, runtime)
    CC2 = Reactant.traced_type_inner(CC, seen, mode, track_numbers, sharding, runtime)
    FC2 = Reactant.traced_type_inner(FC, seen, mode, track_numbers, sharding, runtime)
    CF2 = Reactant.traced_type_inner(CF, seen, mode, track_numbers, sharding, runtime)
    FF2 = Reactant.traced_type_inner(FF2, seen, mode, track_numbers, sharding, runtime)
    FT2 = Base.promote_type(Base.promote_type(Base.promote_type(Base.promote_type(FT2, eltype(CC2)), eltype(FC2)), eltype(CF2)), eltype(FF2))
    return Oceananigans.Grids.OrthogonalSphericalShellGrid{FT2, TX2, TY2, TZ2, Z2, Map2, CC2, FC2, CF2, FF2, Arch}
end

# https://github.com/CliMA/Oceananigans.jl/blob/d9b3b142d8252e8e11382d1b3118ac2a092b38a2/src/ImmersedBoundaries/immersed_boundary_grid.jl#L8
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(OA::Type{ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, Arch}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {FT, TX, TY, TZ, G, I, M, S, Arch}
    TX2 = Reactant.traced_type_inner(TX, seen, mode, track_numbers, sharding, runtime)
    TY2 = Reactant.traced_type_inner(TY, seen, mode, track_numbers, sharding, runtime)
    TZ2 = Reactant.traced_type_inner(TZ, seen, mode, track_numbers, sharding, runtime)
    G2 = Reactant.traced_type_inner(G, seen, mode, track_numbers, sharding, runtime)
    I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)
    M2 = Reactant.traced_type_inner(M, seen, mode, track_numbers, sharding, runtime)
    S2 = Reactant.traced_type_inner(S, seen, mode, track_numbers, sharding, runtime)
    FT2 = eltype(G2)
    return Oceananigans.Grids.OrthogonalSphericalShellGrid{FT2, TX2, TY2, TZ2, G2, I2, M2, S2, Arch}
end

function loop!(model, Ninner)
    Δt = 1200 # 20 minutes
    Oceananigans.TimeSteppers.first_time_step!(model, Δt)
    @trace for _ = 2:Ninner
        Oceananigans.TimeSteppers.time_step!(model, Δt)
    end
    return nothing
end
