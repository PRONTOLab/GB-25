using Reactant

using Enzyme

function double_gyre_model()
    v = Reactant.to_rarray(ones(78, 78, 31))
    pv02 = Reactant.to_rarray(ones(78, 78, 31))
    return (v, pv02)
end

function wind_stress_init()
    res = ones(63, 63)
    res = Reactant.to_rarray(res)
    return res
end

function estimate_tracer_error(v, pv02, wind_stress)    
    u = similar(v, 63, 63, 16)
    fill!(u, 0)
    
    copyto!(@view(v[8:end-8, 8:end-8, 15]), wind_stress)


    pv2 = similar(u, 78, 78, 31)
    fill!(pv2, 0)

    @trace track_numbers=false for _ = 1:3

        copyto!(@view(v[8:end-8, 8:end-8, 8:end-8]), Reactant.Ops.add(v[8:end-8, 8:end-8, 8:end-8], u))

        copyto!(u, v[9:end-7, 7:end-9, 8:end-8])

        copyto!(@view(u[:, :, 2]), Reactant.Ops.add(u[:, :, 2], u[:, :, 8]))

        sVp = Reactant.TracedUtils.broadcast_to_size(v[8:end-8, 8:end-8, 9], size(v[8:end-8, 8:end-8, 8:end-8]))

        copyto!(@view(v[8:end-8, 8:end-8, 8:end-8]), sVp)

        copyto!(@view(pv2[8:end-8, 8:end-8, 8:end-8]), sVp)

    end

    copyto!(pv02, pv2)

    mean_sq_surface_u = sum(u)
    
    return mean_sq_surface_u
end

function estimate_tracer_error(model, wind_stress)
    estimate_tracer_error(model[1], model[2], wind_stress)
end

function differentiate_tracer_error(model, J, dJ)
    v = model[1]
    pv02 = model[2]

    dv = zero(v)
    dpv02 = zero(pv02)

    dedν = autodiff(set_strong_zero(Enzyme.Reverse),
                    estimate_tracer_error, Active,
                    Duplicated(v, dv),
                    Duplicated(pv02, dpv02),
                    Duplicated(J, dJ))

    return dedν, dJ
end

rmodel = double_gyre_model()
rwind_stress = wind_stress_init()

@info "Compiling..."

dJ  = make_zero(rwind_stress) # Field{Face, Center, Nothing}(rmodel.grid)

tic = time()
restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(rmodel, rwind_stress)
rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true differentiate_tracer_error(rmodel, rwind_stress, dJ)
compile_toc = time() - tic

@show compile_toc

i = 10
j = 10

dedν, dJ = rdifferentiate_tracer_error(rmodel, rwind_stress, dJ)

@allowscalar @show dJ[i, j]

# Produce finite-difference gradients for comparison:
ϵ_list = [1e-1, 1e-2, 1e-3] #, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

@allowscalar gradient_list = Array{Float64}[]

for ϵ in ϵ_list
    rmodelP = double_gyre_model()
    rwind_stressP = wind_stress_init()

    @allowscalar diff = 2ϵ * abs(rwind_stressP[i, j])

    @allowscalar rwind_stressP[i, j] = rwind_stressP[i, j] + ϵ * abs(rwind_stressP[i, j])

    sq_surface_uP = restimate_tracer_error(rmodelP, rwind_stressP)

    rmodelM = double_gyre_model()
    rwind_stressM = wind_stress_init()
    @allowscalar rwind_stressM[i, j] = rwind_stressM[i, j] - ϵ * abs(rwind_stressM[i, j])

    sq_surface_uM = restimate_tracer_error(rmodelM, rwind_stressM)

    dsq_surface_u = (sq_surface_uP - sq_surface_uM) / diff

    @show ϵ, dsq_surface_u

end
