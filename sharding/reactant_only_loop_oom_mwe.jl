using Dates

@info "Starting Reactant-only OOM MWE" now(UTC)

ENV["JULIA_DEBUG"] = get(ENV, "JULIA_DEBUG", "Reactant_jll,Reactant")

using Reactant

function allocatorstats()
    device = Reactant.XLA.default_device(Reactant.XLA.default_backend())
    client = Reactant.XLA.platform_name(Reactant.XLA.client(device))
    client == "cpu" && return nothing
    return Reactant.XLA.allocatorstats(device)
end

function maybe_initialize_distributed()
    detector_list = (
        Reactant.Distributed.SlurmEnvDetector(),
        Reactant.Distributed.OpenMPIORTEEnvDetector(),
        Reactant.Distributed.MPIEnvDetector(),
        Reactant.Distributed.OpenMPIPMIXEnvDetector(),
        Reactant.Distributed.GkeTPUCluster(),
        Reactant.Distributed.GceTPUCluster(),
    )

    if !isnothing(findfirst(Reactant.Distributed.is_env_present, detector_list))
        Reactant.Distributed.initialize()
    end
end

function factors(N::Int)
    d = log2(N) / 2
    D = Int(exp2(ceil(Int, d)))

    alternate = 1
    tries = 1
    while N % D != 0
        D -= tries * alternate
        tries += 1
        alternate *= -1
    end

    Dx, Dy = extrema((D, N ÷ D))
    return Dx, Dy
end

@inline halo_wrap(a) = cat(a[:, :, 1:8], a[:, :, 9:1016], a[:, :, 1017:1024]; dims=3)
@inline center_update(a) = cat(a[1:8, :, :], a[9:136, :, :] .+ 1.0, a[137:144, :, :]; dims=1)

function loop_like(
    eta, Ubar, Vbar, eta_f, Ubar_f, Vbar_f,
    u, v, w, T, S, p,
    G1u, G1v, G1T, G1S,
    G2u, G2v, G2T, G2S,
    Ninner,
)
    for _ in 1:Ninner
        u = halo_wrap(center_update(u))
        v = halo_wrap(center_update(v))
        w = halo_wrap(center_update(w))
        T = halo_wrap(center_update(T))
        S = halo_wrap(center_update(S))
        p = halo_wrap(center_update(p))
        G1u = halo_wrap(center_update(G1u))
        G1v = halo_wrap(center_update(G1v))
        G1T = halo_wrap(center_update(G1T))
        G1S = halo_wrap(center_update(G1S))
        G2u = halo_wrap(center_update(G2u))
        G2v = halo_wrap(center_update(G2v))
        G2T = halo_wrap(center_update(G2T))
        G2S = halo_wrap(center_update(G2S))

        eta = eta .+ 1.0
        Ubar = Ubar .+ 1.0
        Vbar = Vbar .+ 1.0
        eta_f = eta_f .+ 1.0
        Ubar_f = Ubar_f .+ 1.0
        Vbar_f = Vbar_f .+ 1.0
    end

    return (
        eta, Ubar, Vbar, eta_f, Ubar_f, Vbar_f,
        u, v, w, T, S, p,
        G1u, G1v, G1T, G1S,
        G2u, G2v, G2T, G2S,
    )
end

maybe_initialize_distributed()

ndevices = length(Reactant.devices())
@info "Visible Reactant devices" ndevices

Dx, Dy = factors(ndevices)
mesh = Sharding.Mesh(reshape(Reactant.devices(), Dx, Dy), (:x, :y))
xyz_sharding = Sharding.NamedSharding(mesh, (nothing, :y, :x))
replicated_sharding = Sharding.NamedSharding(mesh, ())

jobid_procid = string(
    get(ENV, "SLURM_JOB_ID", replace(string(now(UTC)), ':' => '-')),
    ".",
    get(ENV, "SLURM_PROCID", string(getpid())),
)

Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = joinpath(@__DIR__, "mlir_dumps", jobid_procid)
Reactant.Compiler.DEBUG_DISABLE_RESHARDING[] = true
Reactant.Compiler.WHILE_CONCAT[] = true

big_host = fill(1.0, 144, 1536, 1024)
surf_host = fill(1.0, 1, 1536, 1024)

big = Reactant.to_rarray(big_host; sharding=xyz_sharding)
surf = Reactant.to_rarray(surf_host; sharding=xyz_sharding)
Ninner = ConcreteRNumber(2; sharding=replicated_sharding)

@info "Initial allocator stats" allocatorstats()

compiled_loop! = @compile sync=true raise=true loop_like(
    surf, surf, surf, surf, surf, surf,
    big, big, big, big, big, big,
    big, big, big, big,
    big, big, big, big,
    Ninner,
)

@info "Post-compile allocator stats" allocatorstats()

profile_dir = joinpath(@__DIR__, "profiling", jobid_procid, "reactant_only_loop_oom_mwe")
mkpath(profile_dir)

Reactant.with_profiler(profile_dir) do
    @time compiled_loop!(
        surf, surf, surf, surf, surf, surf,
        big, big, big, big, big, big,
        big, big, big, big,
        big, big, big, big,
        Ninner,
    )
end

@info "Final allocator stats" allocatorstats()
@info "Done" now(UTC)
