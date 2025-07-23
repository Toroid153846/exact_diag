#専用の型は考えないという方針で行きます
module ExactDiag
using LinearAlgebra
using PrettyTables

const _dim = Ref(1)
const _site = Ref(1)
function init(dimention::Int, site_number::Int)
  _dim[] = dimention
  _site[] = site_number
end
#進数変換
function Nary_trans(t::Int)
  if t < 0 || t >= _dim[]^_site[]
    throw(ArgumentError("t must be in the range [0, dim^site - 1]"))
  end
  dimention = _dim[]
  site_number = _site[]
  n = Vector{Int}(undef, site_number)
  for i in 1:site_number
    n[site_number+1-i] = t % dimention
    t = div(t, dimention)
  end
  n
end

#進数逆変換
function Nary_reverse(n::Vector{Int})
  t = 0
  c = 1
  dimention = _dim[]
  site_number = _site[]
  for i in 1:site_number
    if n[site_number+1-i] < 0 || n[site_number+1-i] >= dimention
      return -1
    end
    t += n[site_number+1-i] * c
    c *= dimention
  end
  t
end
function spin_op(kind::Char, site::Int, t::Int)
  site_number = _site[]
  if kind == '+'
    n = Nary_trans(t)
    n[(site-1)%site_number+1] += 1
    return (Nary_reverse(n), 1.0 + 0.0im)
  elseif kind == '-'
    n = Nary_trans(t)
    n[(site-1)%site_number+1] -= 1
    return (Nary_reverse(n), 1.0 + 0.0im)
  elseif kind == 'z'
    n = Nary_trans(t)
    return (t, n[(site-1)%site_number+1] % 2 == 0 ? 0.5 + 0.0im : -0.5 + 0.0im)
  else
    throw(ArgumentError("kind must be '+', '-', or 'z'"))
  end
end
function mat_tfim(h_j::ComplexF64)
  dimention = _dim[]
  site_number = _site[]
  mat = zeros(ComplexF64, dimention^site_number, dimention^site_number)
  for t in 0:dimention^site_number-1
    for site in 1:site_number
      (t1, v1) = spin_op('z', site, t)
      (t2, v2) = spin_op('z', site + 1, t)
      (t3, v3) = spin_op('+', site, t)
      (t4, v4) = spin_op('-', site, t)
      if t2 != -1
        mat[t+1, t1+1] += (0.5 + 0.0im) * v1 * v2
      end
      if t3 != -1
        mat[t+1, t3+1] -= (0.5 + 0.0im) * h_j * v3
      end
      if t4 != -1
        mat[t+1, t4+1] -= (0.5 + 0.0im) * h_j * v4
      end
    end
  end
  mat
end
end
using .ExactDiag
using LinearAlgebra
ExactDiag.init(2, 12)
t = 3
n = ExactDiag.Nary_trans(t)
println(n)
n2 = ExactDiag.spin_op('z', 2, t)
println(n2)
mat=ExactDiag.mat_tfim(1.0 + 0.0im)
@time LinearAlgebra.eigen(mat)