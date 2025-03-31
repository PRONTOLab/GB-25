using GordonBell25
using GordonBell25: preamble
using GordonBell25: baroclinic_instability_model
using Oceananigans
using JLD2

using Reactant
using Oceananigans.Architectures: ReactantState
# preamble()

Nt = 10

function benchmark_model(arch, Nx, Ny, Nz)
    model = baroclinic_instability_model(arch, Nx, Ny, Nz; Δt=1)

    if arch isa ReactantState
        _Nt = ConcreteRNumber(Nt)
        @info "Compiling..."
        first! = @compile raise=true sync=true GordonBell25.first_time_step!(model)
        step! = @compile raise=true sync=true GordonBell25.time_step!(model)
        loop! = @compile raise=true sync=true GordonBell25.loop!(model, _Nt)
    else
        _Nt = Nt
        first! = first_time_step!
        step! = time_step!
        loop! = GordonBell25.loop!
    end

    first!(model)
    step!(model)
    step!(model)
    step!(model)

    step_timings = Float64[]
    for trial = 1:6
        _, timing = @timed begin
            for n = 1:10
                step!(model)
            end
        end
        push!(step_timings, timing)
    end

    loop_timings = Float64[]
    for trial = 1:6
        _, timing = @timed loop!(model, Nt)
        push!(loop_timings, timing)
    end

    return step_timings, loop_timings
end


function benchmark_series(arch, filename)
    for FT = (Float32, Float64)
        Oceananigans.defaults.FloatType = FT
        for n = 4:11
            Nx = 2^n
            Ny = 2^(n-1)
            Nz = 128
            @info "Running $Nx, $Ny, $Nz for $FT..."
            step_timings, loop_timings = benchmark_model(arch, Nx, Ny, Nz)
            @show step_timings
            @show loop_timings
    
            addr = string(FT, "/", n)
            file = jldopen(filename, "a+")
            file["$addr/Nx"] = Nx
            file["$addr/Ny"] = Ny
            file["$addr/Nz"] = Nz
            file["$addr/Nt"] = Nt
            file["$addr/n"] = n
            file["$addr/FT"] = FT
            file["$addr/step"] = step_timings
            file["$addr/loop"] = loop_timings
            close(file)
        
            GC.gc(true); GC.gc(false); GC.gc(true)
        end
    end
end

arch = GPU()
filename = "timings_GPU.jld2"
rm(filename, force=true)
benchmark_series(arch, filename)

arch = ReactantState()
filename = "timings_Reactant_GPU.jld2"
rm(filename, force=true)
benchmark_series(arch, filename)

