using Reactant
using OffsetArrays

function comparison_of_views(a, b)

    view(parent(a), 1:1, 1:1, 1:1) .= view(parent(b), 1:1, 1:1, 1:1)
    return a
end

a = zeros(208, 142, 1)
a = OffsetArray(a, -7:200, -22:119, 1:1)

b = zeros(208, 112, 1)
b = OffsetArray(b, -7:200, -7:104, 1:1)

# Making these Reactant arrays (where things break)
a = Reactant.to_rarray(a)
b = Reactant.to_rarray(b)

c = comparison_of_views(a, b)

@info "Done!"