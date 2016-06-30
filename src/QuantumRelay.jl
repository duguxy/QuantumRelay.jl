module QuantumRelay

using SymPy
using Iterators
using PolyLib
using JuMP
using Clp
using LLLplus
using GSL
using PyPlot
using Lora
using Distributions

export
	qrelay_op,
	op_mat,
	scan_maker,
	QRelaySampler

include("util.jl")
include("operators.jl")
include("scan.jl")
include("distributions.jl")

type QRelaySampler
	prob::Function
	psetproposal::Function

	function QRelaySampler{T<:Int}(mat::Array{T, 2}, coef, omega, pdet0)
		U, S, V = PolyLib.smith_normal_form(mat)
		Ui = PolyLib.inverse(U)
		Vi = PolyLib.inverse(V)
		s = diag(S)
		r = countnz(s)
		s0 = s[1:r]
		@assert s0 == ones(r)
		ui1 = Ui[1:r, :]
		ui2 = Ui[r+1:end, :]
		vi1 = Vi[:, 1:r]
		vi2 = Vi[:, r+1:end]
		vi2 = lll(vi2)[1]
		T0 = vi1*ui1
		ui2oc = orthocomp(ui2)
		setc, scan = scan_maker(vi2)

		function prob(na)
		    @assert countnz(ui2*na) == 0
		    b = T0*na
		    setc(-b)
		    total = 0.0
		    for x in Task(scan)
		        nab = vi2*x + b
		        total += prod([c.^complex(n)/factorial(n) for (c, n) in zip(coef, nab)])
		    end
		    return abs(total*omega)^2
		end

		function prob(q, na, mask)
		    q0 = round(Int, q.>0)
		    m0 = round(Int, mask)
		    return prod((q0 + (1-2q0).*pdet0(na)).^m0)
		end

		psetproposal(x::Vector) = QuantumRelay.OrthoNNDist(x, ui2oc)

		new(prob, psetproposal)
	end
end

end