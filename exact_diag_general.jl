module ExactDiag
using LinearAlgebra
using PrettyTables
using SparseArrays     # For eigs function
const _dim = Ref(1)
const _site = Ref(1)
export init, Op, spin_op, id, shift, site_flip, spin_flip, sum_j, block_diag, diag, energy_same_check, energy_gap, block_diag_energy_gap
#region 初期化
function init(dimention::Int, site_number::Int)
  _dim[] = dimention
  _site[] = site_number
end
#endregion

#region 進数変換
function Nary_trans(t::Int)
  if t == -1
    return [-1 for _ in 1:_site[]]
  end
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
#endregion

#region 演算子関係の関数
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
function f_op(f::Function)
  Op(f)
end
function spin_op(kind::Char, site::Int)
  site_number = _site[]
  if kind == '+'
    function plus(t::Int)
      n = Nary_trans(t)
      n[(site-1)%site_number+1] += 1
      return (1.0 + 0.0im, Nary_reverse(n))
    end
    return f_op(plus)
  elseif kind == '-'
    function minus(t::Int)
      n = Nary_trans(t)
      n[(site-1)%site_number+1] -= 1
      return (1.0 + 0.0im, Nary_reverse(n))
    end
    return f_op(minus)
  elseif kind == 'z'
    function z(t::Int)
      n = Nary_trans(t)
      return (n[(site-1)%site_number+1] % 2 == 0 ? 0.5 + 0.0im : -0.5 + 0.0im, t)
    end
    return f_op(z)
  else
    throw(ArgumentError("kind must be '+', '-', or 'z'"))
  end
end
function id(t::Int)
  return (1.0 + 0.0im, t)
end
id() = f_op(id)
function shift(k::Int=1)
  function shift1(t::Int)
    n = Nary_trans(t)
    n1 = circshift(n, k)
    return (1.0 + 0.0im, Nary_reverse(n1))
  end
  return f_op(shift1)
end
function site_flip(t::Int)
  n = Nary_trans(t)
  n1 = reverse(n)
  return (1.0 + 0.0im, Nary_reverse(n1))
end
site_flip() = f_op(site_flip)
function spin_flip(t::Int)
  if _dim[] != 2
    throw(ArgumentError("spin_flip is only defined for dim=2"))
  end
  n = Nary_trans(t)
  n1 = Vector{Int}(undef, _site[])
  for i in 1:_site[]
    if n[i] == 0
      n1[i] = 1
    elseif n[i] == 1
      n1[i] = 0
    else
      throw(ArgumentError("spin_flip is only defined for dim=2"))
    end
  end
  return (1.0 + 0.0im, Nary_reverse(n1))
end
spin_flip() = f_op(spin_flip)
#endregion

#region 演算子の演算
import Base: *, +, -
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
    push!(k, (op21[1] * coeff, op21[2]))
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
function sum_j(mats::Op...)
  ans = mats[1]
  for mat in mats[2:end]
    ans += mat
  end
  ans
end# サイト数以外の任意のiで回す場合にはsum_j(Tuple(f(i) for i in 1:4)...)のように書いて内包表記
function sum_j(f::Function)
  sum_j(Tuple(f(i) for i in 1:_site[])...)
end# サイト数で回す場合にはsum_j(i->f(i))のように書く
#endregion

#region ブロック対角化とその準備
function mat_gene_nonsparse(op1::Union{Matrix{ComplexF64},Op})
  if isa(op1, Matrix{ComplexF64})
    return op1
  end
  dimention = _dim[]
  site_number = _site[]
  mat = zeros(ComplexF64, dimention^site_number, dimention^site_number)
  for t in 0:dimention^site_number-1
    for (coeff, op11) in op1.op
      t1 = t
      coeff1 = coeff
      for op12 in op11
        (v1, t1) = op12(t1)
        coeff1 *= v1
      end
      if t1 >= 0 && t1 < dimention^site_number
        mat[t+1, t1+1] += coeff1
      end
    end
  end
  mat
end
function mat_gene(op1::Union{Matrix{ComplexF64},Op})
  # 既に密行列ならそのまま返す
  if isa(op1, Matrix{ComplexF64})
    return op1
  end

  # パラメータ読み出し
  dim = _dim[]
  nsite = _site[]
  N = dim^nsite

  # スパース用の行・列・値リストを用意
  rows = Int[]
  cols = Int[]
  vals = ComplexF64[]

  # 行列要素を走査
  for t in 0:N-1
    for (coeff, fns) in op1.op
      t1 = t
      c = coeff
      # 各演算子を適用
      for f in fns
        (v, t1) = f(t1)
        c *= v
      end
      # 非ゼロ要素だけを記録
      if 0 ≤ t1 < N
        push!(rows, t + 1)
        push!(cols, t1 + 1)
        push!(vals, c)
      end
    end
  end

  # SparseMatrixCSC を構築して返す
  return sparse(rows, cols, vals, N, N)
end
function nthroots(z::ComplexF64, n::Int)
  if n == 2 && abs(imag(z)) < 1.0e-10 && real(z) > 1.0e-10
    return [-sqrt(real(z)) + 0.0im, sqrt(real(z)) + 0.0im]
  end
  r = abs(z)
  θ = angle(z)
  roots = Vector{ComplexF64}(undef, n)
  for k in 1:n
    if n / k == 2 && abs(imag(z)) < 1.0e-10 && real(z) > 1.0e-10
      roots[k] = -sqrt(real(z)) + 0.0im
      continue
    end
    if n == k
      roots[k] = r^(1 / n) * cis(θ / n)
      continue
    end
    roots[k] = r^(1 / n) * cis(θ / n + 2π * ((n / k)^(-1)))
  end
  return roots
end
function diag(op1::Op)
  mat1 = mat_gene_nonsparse(op1)
  energy = Vector{Float64}()
  energy1, _ = eigen(mat1)
  energy = vcat_energy(energy, energy1)
  return energy
end
function vcat_energy(energy1::Union{Vector{Float64},Vector{ComplexF64}}, energy2::Union{Vector{Float64},Vector{ComplexF64}})
  energy3 = Vector{Float64}()
  for i in energy1
    if abs(imag(i)) > 1.0e-10
      throw(ArgumentError("The input vector contains complex numbers."))
    end
    push!(energy3, real(i))
  end
  for i in energy2
    if abs(imag(i)) > 1.0e-10
      throw(ArgumentError("The input vector contains complex numbers."))
    end
    push!(energy3, real(i))
  end
  return energy3
end
function block_diag(kind::String, op1::Op, op2::Op=id())
  mat1 = mat_gene(op1)
  site_number = _site[]
  dimention = _dim[]
  if kind == "u1"
    energy = Vector{Float64}()
    nn1 = [Vector{Int}() for i in 0:site_number*(dimention-1)]
    for t in 0:dimention^site_number-1
      push!(nn1[sum(Nary_trans(t))+1], t)
    end
    for n2 in nn1
      mat2 = Matrix(mat1[n2.+1, n2.+1])
      if size(mat2, 1) == 1
        push!(energy, real(mat2[1, 1]))
      else
        vals, _ = eigen(mat2)
        energy = vcat_energy(energy, vals)
      end
      # n2_len = length(n2)
      # mat2 = zeros(ComplexF64, n2_len, n2_len)
      # for i in 1:n2_len
      #   for j in 1:n2_len
      #     mat2[i, j] = mat1[n2[i]+1, n2[j]+1]
      #   end
      # end
      # if n2_len == 1
      #   if abs(imag(mat2[1, 1])) > 1.0e-10
      #     throw(ArgumentError("The matrix is not diagonalizable."))
      #   end
      #   push!(energy, real(mat2[1, 1]))
      #   continue
      # end
      # energy1, _ = eigen(mat2)
      # energy = vcat_energy(energy, energy1)
    end
    return energy
  elseif kind == "z2"
    energy = Vector{Float64}()
    nn1 = Vector{Vector{Int}}()
    dim_tot = dimention^site_number
    set_int = Set{Int}()
    count = 0
    for t in 0:dim_tot-1
      if t in set_int
        continue
      end
      count += 1
      push!(nn1, Vector{Int}())
      push!(nn1[count], t)
      push!(set_int, t)
      t1 = copy(t)
      for i in 1:2
        t1 = (op2*t1)[2]
        if t == t1
          break
        end
        push!(nn1[count], t1)
        push!(set_int, t1)
      end
    end
    nthroot1 = nthroots(1.0 + 0.0im, 2)
    nn1_len = length(nn1)
    for i in 1:2
      c = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % 2 != 0
          continue
        end
        c += 1
      end
      mat2 = zeros(ComplexF64, c, c)
      c1 = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % 2 != 0
          continue
        end
        c1 += 1
        c2 = 0
        for k in 1:nn1_len
          if (i * length(nn1[k])) % 2 != 0
            continue
          end
          c2 += 1
          mat2_pat = 0.0 + 0.0im
          n2_len = length(nn1[j])
          for l in 1:n2_len
            mat2_pat += sqrt(length(nn1[k]) / length(nn1[j])) * mat1[nn1[j][l]+1, nn1[k][1]+1] * nthroot1[(i*(l-1)+2-1)%2+1]
          end
          mat2[c1, c2] = mat2_pat
        end
      end
      energy1, _ = eigen(mat2)
      energy = vcat_energy(energy, energy1)
    end
    return energy
  elseif kind == "shift"
    energy = Vector{Float64}()
    nn1 = Vector{Vector{Int}}()
    dim_tot = dimention^site_number
    set_int = Set{Int}()
    count = 0
    for t in 0:dim_tot-1
      if t in set_int
        continue
      end
      count += 1
      push!(nn1, Vector{Int}())
      push!(nn1[count], t)
      push!(set_int, t)
      t1 = copy(t)
      for i in 1:site_number
        t1 = (op2*t1)[2]
        if t == t1
          break
        end
        push!(nn1[count], t1)
        push!(set_int, t1)
      end
    end
    nthroot1 = nthroots(1.0 + 0.0im, site_number)
    nn1_len = length(nn1)
    for i in 1:site_number
      c = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % site_number != 0
          continue
        end
        c += 1
      end
      mat2 = zeros(ComplexF64, c, c)
      c1 = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % site_number != 0
          continue
        end
        c1 += 1
        c2 = 0
        for k in 1:nn1_len
          if (i * length(nn1[k])) % site_number != 0
            continue
          end
          c2 += 1
          mat2_pat = 0.0 + 0.0im
          n2_len = length(nn1[j])
          for l in 1:n2_len
            mat2_pat += sqrt(length(nn1[k]) / length(nn1[j])) * mat1[nn1[j][l]+1, nn1[k][1]+1] * nthroot1[(i*(l-1)+site_number-1)%site_number+1]
          end
          mat2[c1, c2] = mat2_pat
        end
      end
      energy1, _ = eigen(mat2)
      energy = vcat_energy(energy, energy1)
    end
    return energy
  elseif kind == "u1shift"
    energy = Vector{Float64}()
    nn1 = [Vector{Int}() for i in 0:site_number*(dimention-1)]
    for t in 0:dimention^site_number-1
      push!(nn1[sum(Nary_trans(t))+1], t)
    end
    for n2 in nn1
      n2_len = length(n2)
      mat2 = zeros(ComplexF64, n2_len, n2_len)
      for i in 1:n2_len
        for j in 1:n2_len
          mat2[i, j] = mat1[n2[i]+1, n2[j]+1]
        end
      end
      if n2_len == 1
        if abs(imag(mat2[1, 1])) > 1.0e-10
          throw(ArgumentError("The matrix is not diagonalizable."))
        end
        push!(energy, real(mat2[1, 1]))
        continue
      end
      nn2 = Vector{Vector{Int}}()
      dim_tot = dimention^site_number
      set_int = Set{Int}()
      count = 0
      for t in n2
        if t in set_int
          continue
        end
        count += 1
        push!(nn2, Vector{Int}())
        push!(nn2[count], t)
        push!(set_int, t)
        t1 = copy(t)
        for i in 1:site_number
          t1 = (op2*t1)[2]
          if t == t1
            break
          end
          push!(nn2[count], t1)
          push!(set_int, t1)
        end
      end
      nthroot1 = nthroots(1.0 + 0.0im, site_number)
      nn2_len = length(nn2)
      for i in 1:site_number
        c = 0
        for j in 1:nn2_len
          if (i * length(nn2[j])) % site_number != 0
            continue
          end
          c += 1
        end
        mat3 = zeros(ComplexF64, c, c)
        c1 = 0
        for j in 1:nn2_len
          if (i * length(nn2[j])) % site_number != 0
            continue
          end
          c1 += 1
          c2 = 0
          for k in 1:nn2_len
            if (i * length(nn2[k])) % site_number != 0
              continue
            end
            c2 += 1
            mat3_pat = 0.0 + 0.0im
            n2_len = length(nn2[j])
            for l in 1:n2_len
              mat3_pat += sqrt(length(nn2[k]) / length(nn2[j])) * mat1[nn2[j][l]+1, nn2[k][1]+1] * nthroot1[(i*(l-1)+site_number-1)%site_number+1]
            end
            mat3[c1, c2] = mat3_pat
          end
        end
        energy1, _ = eigen(mat3)
        energy = vcat_energy(energy, energy1)
      end
    end
    return energy
  elseif kind == "shiftz2"
    energy = Vector{Float64}()
    nn1 = Vector{Vector{Int}}()
    dim_tot = dimention^site_number
    set_int = Set{Int}()
    count = 0
    for t in 0:dim_tot-1
      if t in set_int
        continue
      end
      count += 1
      push!(nn1, Vector{Int}())
      push!(nn1[count], t)
      push!(set_int, t)
      t1 = copy(t)
      for i in 1:site_number*2
        t1 = (op2*t1)[2]
        if t == t1
          break
        end
        push!(nn1[count], t1)
        push!(set_int, t1)
      end
    end
    nthroot1 = nthroots(1.0 + 0.0im, site_number * 2)
    nn1_len = length(nn1)
    for i in 1:(site_number*2)
      c = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % (site_number * 2) != 0
          continue
        end
        c += 1
      end
      mat2 = zeros(ComplexF64, c, c)
      c1 = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % (site_number * 2) != 0
          continue
        end
        c1 += 1
        c2 = 0
        for k in 1:nn1_len
          if (i * length(nn1[k])) % (site_number * 2) != 0
            continue
          end
          c2 += 1
          mat2_pat = 0.0 + 0.0im
          n2_len = length(nn1[j])
          for l in 1:n2_len
            mat2_pat += sqrt(length(nn1[k]) / length(nn1[j])) * mat1[nn1[j][l]+1, nn1[k][1]+1] * nthroot1[(i*(l-1)+(site_number*2)-1)%(site_number*2)+1]
          end
          mat2[c1, c2] = mat2_pat
        end
      end
      energy1, _ = eigen(mat2)
      energy = vcat_energy(energy, energy1)
    end
    return energy
  else
    throw(ArgumentError("kind must be 'u1', 'z2', 'shift', 'u1shift', or 'shiftz2'"))
  end
end
#endregion

function energy_same_check(energy1::Vector{Float64}, energy2::Vector{Float64})
  if length(energy1) != length(energy2)
    throw(ArgumentError("The lengths of the two energy vectors are not equal."))
  end
  sorted_energy1 = sort(energy1)
  sorted_energy2 = sort(energy2)
  for i in 1:length(energy1)
    if abs(sorted_energy1[i] - sorted_energy2[i]) > 1.0e-10
      return false
    end
  end
  return true
end
function block_diag_g(kind::String, ham::Union{Matrix{ComplexF64},Op}, v1::Vector{Int}=[i for i in 1:_dim[]^_site[]])
  return diag(ham)
end
function block_diag_g(kind::String, ham::Union{Matrix{ComplexF64},Op}, v1::Vector{Int}=[i for i in 1:_dim[]^_site[]], ops::Tuple{Op,Function}...)
  mat1 = mat_gene(op1)
  site_number = _site[]
  dimention = _dim[]
  if kind == "u1"
    energy = Vector{Float64}()
    nn1 = [Vector{Int}() for i in 0:site_number*(dimention-1)]
    for t in 0:dimention^site_number-1
      push!(nn1[sum(Nary_trans(t))+1], t)
    end
    for n2 in nn1
      n2_len = length(n2)
      mat2 = zeros(ComplexF64, n2_len, n2_len)
      for i in 1:n2_len
        for j in 1:n2_len
          mat2[i, j] = mat1[n2[i]+1, n2[j]+1]
        end
      end
      if n2_len == 1
        if abs(imag(mat2[1, 1])) > 1.0e-10
          throw(ArgumentError("The matrix is not diagonalizable."))
        end
        push!(energy, real(mat2[1, 1]))
        continue
      end
      energy1 = block_diag_g("else", mat2, n2, ops[2:end]...)
      energy = vcat_energy(energy, energy1)
    end
    return energy
  elseif kind == "else"
  else
    throw(ArgumentError("kind must be 'u1' or 'else'"))
  end
end
function energy_reshape(energys::Union{Vector{Float64},Vector{ComplexF64}})
  if isa(energys, Vector{Float64})
    return energys
  elseif isa(energys, Vector{ComplexF64})
    energys1 = Vector{Float64}(undef, length(energys))
    for i in eachindex(energys)
      if abs(imag(energys[i])) > 1.0e-10
        throw(ArgumentError("The input vector contains complex numbers."))
      end
      energys1[i] = real(energys[i])
    end
    return energys1
  else
    throw(ArgumentError("energys must be a Vector of Float64 or ComplexF64"))
  end
end
function energy_gap(energys::Union{Vector{Float64},Vector{ComplexF64}})
  if length(energys) < 3
    throw(ArgumentError("energys must contain at least three elements"))
  end
  energys1 = energy_reshape(energys)
  sorted_energies = sort(energys1)
  s = 0.0
  for i in 1:length(sorted_energies)-2
    a = sorted_energies[i+1] - sorted_energies[i]
    b = sorted_energies[i+2] - sorted_energies[i+1]
    s += min(a, b) / max(a, b)
  end
  return s / (length(sorted_energies) - 2)
end
function has_duplicates(v::AbstractVector)
  seen = Set{eltype(v)}()
  for x in v
    if x in seen
      return true
    end
    push!(seen, x)
  end
  return false
end
function block_diag_energy_gap(kind::String, op1::Op, op2::Op=id())
  mat1 = mat_gene(op1)
  site_number = _site[]
  dimention = _dim[]
  if kind == "u1"
    energy = Vector{Float64}()
    nn1 = [Vector{Int}() for i in 0:site_number*(dimention-1)]
    for t in 0:dimention^site_number-1
      push!(nn1[sum(Nary_trans(t))+1], t)
    end
    for n2 in nn1
      mat2 = Matrix(mat1[n2.+1, n2.+1])
      if size(mat2, 1) == 1
        push!(energy, real(mat2[1, 1]))
      else
        vals, _ = eigen(mat2)
        energy = vcat_energy(energy, vals)
      end
      # n2_len = length(n2)
      # mat2 = zeros(ComplexF64, n2_len, n2_len)
      # for i in 1:n2_len
      #   for j in 1:n2_len
      #     mat2[i, j] = mat1[n2[i]+1, n2[j]+1]
      #   end
      # end
      # if n2_len == 1
      #   if abs(imag(mat2[1, 1])) > 1.0e-10
      #     throw(ArgumentError("The matrix is not diagonalizable."))
      #   end
      #   push!(energy, real(mat2[1, 1]))
      #   continue
      # end
      # energy1, _ = eigen(mat2)
      # energy = vcat_energy(energy, energy1)
    end
    return energy
  elseif kind == "z2"
    energy = Vector{Float64}()
    nn1 = Vector{Vector{Int}}()
    dim_tot = dimention^site_number
    set_int = Set{Int}()
    count = 0
    for t in 0:dim_tot-1
      if t in set_int
        continue
      end
      count += 1
      push!(nn1, Vector{Int}())
      push!(nn1[count], t)
      push!(set_int, t)
      t1 = copy(t)
      for i in 1:2
        t1 = (op2*t1)[2]
        if t == t1
          break
        end
        push!(nn1[count], t1)
        push!(set_int, t1)
      end
    end
    nthroot1 = nthroots(1.0 + 0.0im, 2)
    nn1_len = length(nn1)
    for i in 1:2
      c = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % 2 != 0
          continue
        end
        c += 1
      end
      mat2 = zeros(ComplexF64, c, c)
      c1 = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % 2 != 0
          continue
        end
        c1 += 1
        c2 = 0
        for k in 1:nn1_len
          if (i * length(nn1[k])) % 2 != 0
            continue
          end
          c2 += 1
          mat2_pat = 0.0 + 0.0im
          n2_len = length(nn1[j])
          for l in 1:n2_len
            mat2_pat += sqrt(length(nn1[k]) / length(nn1[j])) * mat1[nn1[j][l]+1, nn1[k][1]+1] * nthroot1[(i*(l-1)+2-1)%2+1]
          end
          mat2[c1, c2] = mat2_pat
        end
      end
      energy1, _ = eigen(mat2)
      energy = vcat_energy(energy, energy1)
    end
    return energy
  elseif kind == "shift"
    energy = Vector{Float64}()
    nn1 = Vector{Vector{Int}}()
    dim_tot = dimention^site_number
    set_int = Set{Int}()
    count = 0
    for t in 0:dim_tot-1
      if t in set_int
        continue
      end
      count += 1
      push!(nn1, Vector{Int}())
      push!(nn1[count], t)
      push!(set_int, t)
      t1 = copy(t)
      for i in 1:site_number
        t1 = (op2*t1)[2]
        if t == t1
          break
        end
        push!(nn1[count], t1)
        push!(set_int, t1)
      end
    end
    nthroot1 = nthroots(1.0 + 0.0im, site_number)
    nn1_len = length(nn1)
    for i in 1:site_number
      c = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % site_number != 0
          continue
        end
        c += 1
      end
      mat2 = zeros(ComplexF64, c, c)
      c1 = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % site_number != 0
          continue
        end
        c1 += 1
        c2 = 0
        for k in 1:nn1_len
          if (i * length(nn1[k])) % site_number != 0
            continue
          end
          c2 += 1
          mat2_pat = 0.0 + 0.0im
          n2_len = length(nn1[j])
          for l in 1:n2_len
            mat2_pat += sqrt(length(nn1[k]) / length(nn1[j])) * mat1[nn1[j][l]+1, nn1[k][1]+1] * nthroot1[(i*(l-1)+site_number-1)%site_number+1]
          end
          mat2[c1, c2] = mat2_pat
        end
      end
      energy1, _ = eigen(mat2)
      energy = vcat_energy(energy, energy1)
    end
    return energy
  elseif kind == "u1shift"
    energy_gap_vector = Vector{Tuple{Int,Float64}}()
    nn1 = [Vector{Int}() for i in 0:site_number*(dimention-1)]
    for t in 0:dimention^site_number-1
      push!(nn1[sum(Nary_trans(t))+1], t)
    end
    for n2 in nn1
      n2_len = length(n2)
      mat2 = zeros(ComplexF64, n2_len, n2_len)
      for i in 1:n2_len
        for j in 1:n2_len
          mat2[i, j] = mat1[n2[i]+1, n2[j]+1]
        end
      end
      if n2_len == 1
        if abs(imag(mat2[1, 1])) > 1.0e-10
          throw(ArgumentError("The matrix is not diagonalizable."))
        end
        continue
      end
      nn2 = Vector{Vector{Int}}()
      dim_tot = dimention^site_number
      set_int = Set{Int}()
      count = 0
      for t in n2
        if t in set_int
          continue
        end
        count += 1
        push!(nn2, Vector{Int}())
        push!(nn2[count], t)
        push!(set_int, t)
        t1 = copy(t)
        for i in 1:site_number
          t1 = (op2*t1)[2]
          if t == t1
            break
          end
          push!(nn2[count], t1)
          push!(set_int, t1)
        end
      end
      nthroot1 = nthroots(1.0 + 0.0im, site_number)
      nn2_len = length(nn2)
      for i in 1:site_number
        c = 0
        for j in 1:nn2_len
          if (i * length(nn2[j])) % site_number != 0
            continue
          end
          c += 1
        end
        mat3 = zeros(ComplexF64, c, c)
        c1 = 0
        for j in 1:nn2_len
          if (i * length(nn2[j])) % site_number != 0
            continue
          end
          c1 += 1
          c2 = 0
          for k in 1:nn2_len
            if (i * length(nn2[k])) % site_number != 0
              continue
            end
            c2 += 1
            mat3_pat = 0.0 + 0.0im
            n2_len = length(nn2[j])
            for l in 1:n2_len
              mat3_pat += sqrt(length(nn2[k]) / length(nn2[j])) * mat1[nn2[j][l]+1, nn2[k][1]+1] * nthroot1[(i*(l-1)+site_number-1)%site_number+1]
            end
            mat3[c1, c2] = mat3_pat
          end
        end
        energy1, _ = eigen(mat3)
        if length(energy1) < 3 || has_duplicates(energy1) == true
          continue
        end
        if isnan(energy_gap(energy1))
          continue
        end
        push!(energy_gap_vector, (length(energy1), energy_gap(energy1)))
      end
    end
    return energy_gap_vector
  elseif kind == "shiftz2"
    energy_gap_vector = Vector{Tuple{Int,Float64}}()
    nn1 = Vector{Vector{Int}}()
    dim_tot = dimention^site_number
    set_int = Set{Int}()
    count = 0
    for t in 0:dim_tot-1
      if t in set_int
        continue
      end
      count += 1
      push!(nn1, Vector{Int}())
      push!(nn1[count], t)
      push!(set_int, t)
      t1 = copy(t)
      for i in 1:site_number*2
        t1 = (op2*t1)[2]
        if t == t1
          break
        end
        push!(nn1[count], t1)
        push!(set_int, t1)
      end
    end
    nthroot1 = nthroots(1.0 + 0.0im, site_number * 2)
    nn1_len = length(nn1)
    for i in 1:(site_number*2)
      c = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % (site_number * 2) != 0
          continue
        end
        c += 1
      end
      mat2 = zeros(ComplexF64, c, c)
      c1 = 0
      for j in 1:nn1_len
        if (i * length(nn1[j])) % (site_number * 2) != 0
          continue
        end
        c1 += 1
        c2 = 0
        for k in 1:nn1_len
          if (i * length(nn1[k])) % (site_number * 2) != 0
            continue
          end
          c2 += 1
          mat2_pat = 0.0 + 0.0im
          n2_len = length(nn1[j])
          for l in 1:n2_len
            mat2_pat += sqrt(length(nn1[k]) / length(nn1[j])) * mat1[nn1[j][l]+1, nn1[k][1]+1] * nthroot1[(i*(l-1)+(site_number*2)-1)%(site_number*2)+1]
          end
          mat2[c1, c2] = mat2_pat
        end
      end
      energy1, _ = eigen(mat2)
      if length(energy1) < 3 || has_duplicates(energy1) == true
        continue
      end
      if isnan(energy_gap(energy1))
        continue
      end
      push!(energy_gap_vector, (length(energy1), energy_gap(energy1)))
    end
    return energy_gap_vector
  else
    throw(ArgumentError("kind must be 'u1', 'z2', 'shift', 'u1shift', or 'shiftz2'"))
  end
end
end
# using .ExactDiag
# Δ = 1.0
# n = 13
# init(2, n)
# H = sum_j(j -> 0.25 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)) + 0.5 * Δ * spin_op('z', j) * spin_op('z', j + 1))
# @time block_diag("u1", H)
# @time block_diag("shift", H, shift())
# @time block_diag("u1shift", H, shift())
# @time block_diag("shiftz2", H, shift() * spin_flip())
# # println(energy_same_check(diag(H), block_diag("u1shift", H, shift())))
# # println(energy_same_check(diag(H), block_diag("shiftz2", H, shift() * spin_flip())))
# println("")
using .ExactDiag
using UnionFind
using LinearAlgebra
L = 14
hj = 1.0
Δ = 2.0
init(2, L)
sym = "u1"
uf = UnionFinder(8)
H1 = sum_j(j -> 0.5 * spin_op('z', j) * spin_op('z', j + 1) - 0.5 * hj * (spin_op('+', j) + spin_op('-', j)))
H2 = sum_j(j -> 0.25 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)) + 0.5 * Δ * spin_op('z', j) * spin_op('z', j + 1))
H3 = -1.0 / 2.0 * sum_j(j -> -0.5 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)))
S_z = sum_j(j -> spin_op('z', j))
@time block_diag("u1", H3)
@time block_diag("u1shift", H3, shift())
using Plots
using Statistics
x = Vector{Int}()
y1 = Vector{Float64}()
se1 = Vector{Float64}()

y2 = Vector{Float64}()
se2 = Vector{Float64}()
y3 = Vector{Float64}()
for L in 6:16
  println("L = ", L)
  push!(x, L)
  hj = 1.0
  Δ = 2.0
  init(2, L)
  H1 = sum_j(j -> 0.5 * spin_op('z', j) * spin_op('z', j + 1) - 0.5 * hj * (spin_op('+', j) + spin_op('-', j)))
  H2 = sum_j(j -> 0.25 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)) + 0.5 * Δ * spin_op('z', j) * spin_op('z', j + 1))
  H3 = -1.0 / 2.0 * sum_j(j -> -0.5 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)))
  a1 = block_("shiftz2", H1, shift() * spin_flip())
  ma = 0
  for i in eachindex(a1)
    ma = max(a1[i][1], ma)
  end
  for i in eachindex(a1)
    if a1[i][1] == ma
      push!(y1, a1[i][2])
      break
    end
  end
  a2 = block_diag_energy_gap("u1shift", H2, shift())
  ma = 0
  for i in eachindex(a2)
    ma = max(a2[i][1], ma)
  end
  for i in eachindex(a2)
    if a2[i][1] == ma
      push!(y2, a2[i][2])
      break
    end
  end
  a3 = block_diag_energy_gap("u1shift", H3, shift())
  ma = 0
  for i in eachindex(a3)
    ma = max(a3[i][1], ma)
  end
  for i in eachindex(a3)
    if a3[i][1] == ma
      push!(y3, a3[i][2])
      break
    end
  end
end
plot(
  x, y1;
  # yerror    = se1,
  marker=:circle,
  label="transverse magnetic field ising model",
)
plot!(
  x, y2;
  # yerror    = se2,
  marker=:diamond,
  label="XXZ model",
)
plot!(
  x, y3;
  # yerror    = se2,
  marker=:square,
  label="XY model",
)
xlabel!("site number")
ylabel!("rvalue")
title!("site number vs rvalue")

