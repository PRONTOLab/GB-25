function sync_states!(r_model, c_model)
    cu = c_model.velocities.u
    cv = c_model.velocities.v
    ru = r_model.velocities.u
    rv = r_model.velocities.v

    copyto!(parent(ru), Reactant.to_rarray(parent(cu)))
    copyto!(parent(rv), Reactant.to_rarray(parent(cv)))

    r_tracers = r_model.tracers
    c_tracers = c_model.tracers
    for (rc, cc) in zip(r_tracers, c_tracers)
        copyto!(parent(rc), Reactant.to_rarray(parent(cc)))
    end

    return nothing
end
