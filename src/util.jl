export
    orthocomp,
    pdet_maker,
    bin2

function orthocomp(m)
    U, S, V = PolyLib.smith_normal_form(m)
    return lll(PolyLib.inverse(V)[:, size(m)[1]+1:end])[1]
end

nonneg(v) = all(v.>=0) ? true : false

function unequal_sample_maker(p)
    perm = sortperm(p, rev=true)
    p0 = p[perm]
    for i=1:length(p0)-1
        p0[i+1] += p0[i]
    end

    function sample()
        r = rand()
        for i=1:length(p0)
            if r < p0[i]
                return perm[i]
            end
        end
    end
end

function pdet_maker(eta, pdc)
    a = 1 - pdc
    b = 1 - eta.*a
    function pdet0(i)
        return a .* b.^i
    end
    return pdet0
end

function bin2(arr)
    n1 = length(arr)
    n2 = div(n1, 2)
    arr2 = zeros(n2)
    for i = 1:n2
        arr2[i] = (arr[2i-1] + arr[2i]) / 2
    end
    return arr2
end