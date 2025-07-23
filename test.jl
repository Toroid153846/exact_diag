module ExactDiag
using LinearAlgebra
using PrettyTables
using SparseArrays
using UnionFind
using Printf
const _dim = Ref(1)
const _site = Ref(1)
const _trans = Ref(Vector{Vector{Int}}())
const _reverse = Ref(Dict{Vector{Int},Int}())
export _dim, _site, init, Op, id, shift, spin_flip, site_flip, spin, matrix# ここに外でも使う関数を列挙する
function init(dim::Int, site::Int)
  _dim[] = dim
  _site[] = site
  dim_tot = dim^site
  _trans[] = Vector{Vector{UInt8}}(undef, dim_tot)
  row = zeros(Int, site)
  _trans[][1] = copy(row)
  @inbounds for i in 1:dim_tot-1
    j = site
    while true
      v = row[j] + 1
      if v == dim
        row[j] = 0
        j -= 1
      else
        row[j] = v
        break
      end
    end
    _trans[][i+1] = copy(row)
  end
  _reverse[] = Dict{Vector{Int},Int}(_trans[][i] => i for i in eachindex(_trans[]))
end
function Nary_reverse(n::Vector{Int})
  if haskey(_reverse[], n)
    return copy(_reverse[][n])
  elseif length(n) != _site[]
    throw(ArgumentError("n must be a vector of length site"))
  else
    throw(ArgumentError("n is not a valid state"))
  end
end
function Nary_trans(t::Int)
  if t == 0
    return [-1 for _ in 1:_site[]]
  elseif t < 1 || t > _dim[]^_site[]
    throw(ArgumentError("t must be in the range [0, dim^site - 1]"))
  else
    return copy(_trans[][t])
  end
end
struct Op
  op::Vector{Tuple{ComplexF64,Vector{Function}}}
  function Op(op::Vector{Tuple{ComplexF64,Vector{Function}}})
    new(op)
  end
  function Op(op1::Function)
    vecf = Vector{Function}([op1])
    Op([(1.0 + 0.0im, vecf)])
  end
end
import Base: *, +, -, show, sum
function +(ops::Op...)
  k = Vector{Tuple{ComplexF64,Vector{Function}}}()
  for op1 in ops
    k = vcat(k, op1.op)
  end
  Op(k)
end
function *(op1::Op)
  op1
end
function *(op1::Op, op2::Op...)
  op3 = *(op2...)
  k = Vector{Tuple{ComplexF64,Vector{Function}}}()
  for op11 in op1.op
    for op31 in op3.op
      push!(k, (op11[1] * op31[1], vcat(op11[2], op31[2])))
    end
  end
  Op(k)
end
function *(coeff::Union{ComplexF64,Float64}, op1::Op...)
  op2 = *(op1...)
  k = Vector{Tuple{ComplexF64,Vector{Function}}}()
  for op21 in op2.op
    push!(k, (op21[1] * ComplexF64(coeff), op21[2]))
  end
  Op(k)
end
function *(op1::Op, t::Int)
  t1 = copy(t)
  for op11 in op1.op[1][2]
    t1 = op11(t1)[2]
  end
  sum = 0.0 + 0.0im
  for op11 in op1.op
    product = op11[1]
    t2 = copy(t)
    for op12 in op11[2]
      t2 = op12(t2)[2]
      product *= op12(t2)[1]
    end
    if t2 != t1
      throw(ArgumentError("The operator does not preserve the state."))
    end
    sum += product
  end
  return (sum, t1)
end
function -(op1::Op)
  (-1.0 + 0.0im) * op1
end
function -(op1::Op, op2::Op)
  op1 + (-op2)
end
function -(op1::Op, op2::Op...)
  op3 = +(op2...)
  op1 - op3
end
function sum(mats::Op...)
  ans = mats[1]
  for mat in mats[2:end]
    ans += mat
  end
  ans
end
function sum(f::Function, k::Int=0)
  sum_j(Tuple(f(i) for i in 1:(_site[]-k))...)
end
function id(i::Int)
  if i < 1 || i > _dim[]^_site[]
    throw(ArgumentError("i must be in the range [1, dim^site]"))
  end
  return (1.0 + 0.0im, i)
end
id() = Op(id)
function shift(k::Int=1)
  function shift1(t::Int)
    n = Nary_trans(t)
    n1 = circshift(n, k)
    return (1.0 + 0.0im, Nary_reverse(n1))
  end
  return Op(shift1)
end
function site_flip(t::Int)
  n = Nary_trans(t)
  n1 = reverse(n)
  return (1.0 + 0.0im, Nary_reverse(n1))
end
site_flip() = Op(site_flip)
function spin_flip(t::Int)
  if _dim[] != 2
    throw(ArgumentError("spin_flip is only defined for dim=2"))
  end
  n = Nary_trans(t)
  n1 = Vector{Int}(undef, _site[])
  for i in 1:_site[]
    n1[i] = 1 - n[i]
  end
  return (1.0 + 0.0im, Nary_reverse(n1))
end
spin_flip() = Op(spin_flip)
function spin(kind::Char, site::Int)
  site_number = _site[]
  idx = (site - 1) % site_number + 1
  if kind == '+'
    plus = function (t::Int)
      n = Nary_trans(t)
      n[idx] += 1
      return n[idx] < _dim[] ? (1.0 + 0im, Nary_reverse(n)) : (0.0 + 0im, t)
    end
    return Op(plus)
  elseif kind == '-'
    minus = function (t::Int)
      n = Nary_trans(t)
      n[idx] -= 1
      return n[idx] > -1 ? (1.0 + 0im, Nary_reverse(n)) : (0.0 + 0im, t)
    end
    return Op(minus)
  elseif kind == 'x'
    plus = function (t::Int)
      n = Nary_trans(t)
      n[idx] += 1
      n[idx] < _dim[] ? (1.0 + 0im, Nary_reverse(n)) : (0.0 + 0im, t)
    end
    minus = function (t::Int)
      n = Nary_trans(t)
      n[idx] -= 1
      n[idx] > -1 ? (1.0 + 0im, Nary_reverse(n)) : (0.0 + 0im, t)
    end
    return 0.5 * (Op(plus) + Op(minus))
  elseif kind == 'y'
    plus = function (t::Int)
      n = Nary_trans(t)
      n[idx] += 1
      n[idx] < _dim[] ? (1.0 + 0im, Nary_reverse(n)) : (0.0 + 0im, t)
    end
    minus = function (t::Int)
      n = Nary_trans(t)
      n[idx] -= 1
      n[idx] > -1 ? (1.0 + 0im, Nary_reverse(n)) : (0.0 + 0im, t)
    end
    return (-0.5im) * (Op(plus) - Op(minus))
  elseif kind == 'z'
    z = function (t::Int)
      n = Nary_trans(t)
      ((_dim[] - 1.0) / 2.0 - n[idx] + 0.0im, t)
    end
    return Op(z)
  else
    throw(ArgumentError("kind must be '+', '-', 'x', 'y', or 'z'"))
  end
end
function matrix(op1::Union{Matrix{ComplexF64},Op})
  if isa(op1, Matrix{ComplexF64})
    return op1
  end
  dimention = _dim[]
  site_number = _site[]
  dim_tot = dimention^site_number
  mat = zeros(ComplexF64, dim_tot, dim_tot)
  for t in 1:dim_tot
    for (coeff, op11) in op1.op
      t1 = t
      coeff1 = coeff
      for op12 in op11
        (v1, t1) = op12(t1)
        coeff1 *= v1
      end
      mat[t, t1] += coeff1
    end
  end
  return mat
end
function complex_formatter(; digits::Int=1)
  return (v, i, j) -> begin
    if v == 0 + 0im
      @sprintf("%.*f", digits, 0.0)
    elseif isa(v, Complex)
      rea = round(real(v), digits=digits)
      image = round(imag(v), digits=digits)
      if image == 0
        @sprintf("%.*f", digits, rea)
      elseif rea == 0
        @sprintf("%.*fim", digits, image)
      else
        sign = image > 0 ? "+" : "-"
        @sprintf("%.*f%s%.*fim", digits, rea, sign, digits, abs(image))
      end
    elseif isa(v, Number)
      @sprintf("%.*f", digits, v)
    else
      string(v)
    end
  end
end
function show(op1::Op, digit::Int=1)
  mat1 = matrix(op1)
  pretty_table(mat1, header=([join(string.(Nary_trans(i)), "") for i in 1:size(mat1, 1)]), row_labels=([join(string.(Nary_trans(i)), "") for i in 1:size(mat1, 2)]), formatters=complex_formatter(digits=digit))
end
end
