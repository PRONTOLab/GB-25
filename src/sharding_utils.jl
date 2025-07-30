using Reactant
using Dates

struct TreeSharding{S} <: Sharding.AbstractSharding
    sharding::S
end

Sharding.is_sharded(sharding::TreeSharding) = true

Sharding.ndevices(sharding::TreeSharding) = Sharding.ndevices(sharding.sharding)
Sharding.shard_type(::Type{TreeSharding{S}}, N) where {S} = Sharding.shard_type(S, N)

Base.getproperty(t::TreeSharding, x) = t
function Base.getproperty(t::TreeSharding, x::Symbol)
    x == :sharding && return getfield(t, :sharding)
    return t
end

function (sharding::TreeSharding)(
    client::Reactant.XLA.AbstractClient, device, x::Union{AbstractArray,Number}
)
    return sharding.sharding(client, device, x)
end

"""
    factors(N::Int) -> NTuple{2, Int}

Determine two adjectent factors of `N`, useful for finding partitioning factors for sharding.
"""
function factors(N::Int)
    d = log2(N) / 2
    D = Int(exp2(ceil(Int, d)))

    alternate = 1
    tries = 1
    while (N % D != 0)
        D -= tries * alternate
        tries += 1
        alternate *= -1
    end

    # Always make Dy >= Dx
    Dx, Dy = extrema((D, N ÷ D))

    Dx * Dy != N && error("The product $(Dx) * $(Dy) is not equal to the input argument $(N), there is a bug in this function!")

    return Dx, Dy
end

function allocatorstats()
    device = Reactant.XLA.default_device(Reactant.XLA.default_backend())
    client = Reactant.XLA.platform_name(Reactant.XLA.client(device))
    if client == "cpu"
        nothing
    else
        Reactant.XLA.allocatorstats(device)
    end
end

function initialize(; kwargs...)
    # TODO: improve the condition by checking the device we're on?
    if !(get(ENV, "CI", "false") == "true" || contains(get(ENV, "XLA_FLAGS", ""), "--xla_force_host_platform_device_count"))
        Reactant.Distributed.initialize(; kwargs...)
    end
end

get_jobid_procid() =
    string(
        get(ENV, "SLURM_JOB_ID", replace(string(now(UTC)), ':' => '-')),
        ".",
        get(ENV, "SLURM_PROCID", string(getpid()))
    )

"""
    is_distributed_env_present() :: Bool

Return whether a distributed environment (e.g. Slurm, OpenMPI, generic MPI, Gke TPU, Gce TPU, etc...) is detected.
"""
function is_distributed_env_present()
    # Adapted from https://github.com/EnzymeAD/Reactant.jl/blob/7bbd7ae5d88151a1a966d257e414c9c2cbc1fd84/src/Distributed.jl#L98
    detector_list= [
        Reactant.Distributed.SlurmEnvDetector(),
        Reactant.Distributed.OpenMPIORTEEnvDetector(),
        Reactant.Distributed.MPIEnvDetector(),
        # Keep this at the end since parsing for this is a bit flaky
        Reactant.Distributed.OpenMPIPMIXEnvDetector(),
        # Cloud TPU environments
        Reactant.Distributed.GkeTPUCluster(),
        Reactant.Distributed.GceTPUCluster(),
    ]

    return !isnothing(findfirst(Reactant.Distributed.is_env_present, detector_list))
end
