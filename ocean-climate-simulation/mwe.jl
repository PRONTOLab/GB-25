using Reactant

addone!(data, vec) = data .+= vec

function loop!(data, Nt)
    @trace for n = 1:Nt
        addone!(data)
    end
    return nothing
end

data = Reactant.to_rarray(zeros(1))
Nt = ConcreteRNumber(10)
rloop! = @compile sync=true raise=true loop!(data, Nt)
rloop!(data, Nt)

