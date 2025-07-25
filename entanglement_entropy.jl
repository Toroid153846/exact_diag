# ここでは演算子Opを定義して祖行列を用いて厳密対角化の関数を再帰的に定義します。
module ExactDiag
using LinearAlgebra
using PrettyTables
using SparseArrays     # For eigs function
using UnionFind
const _dim = Ref(1)
const _site = Ref(1)
export init, Op, spin_op, id, shift, site_flip, spin_flip, sum_j, block_diag, block_diag_energy, true_f, mat_gene_nonsparse, shift_z2, energy_same_check, block_diag_energy_shiftz2, entanglement_entropy_show, block_diag_entropy, sum_j_fixed, block_diag_entropy_z2, block_diag_entropy_u1z2
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
  elseif kind == 'x'
    function plus(t::Int)
      n = Nary_trans(t)
      n[(site-1)%site_number+1] += 1
      return (1.0 + 0.0im, Nary_reverse(n))
    end
    function minus(t::Int)
      n = Nary_trans(t)
      n[(site-1)%site_number+1] -= 1
      return (1.0 + 0.0im, Nary_reverse(n))
    end
    return f_op(plus) + f_op(minus)
  elseif kind == 'y'
    function plus(t::Int)
      n = Nary_trans(t)
      n[(site-1)%site_number+1] += 1
      return (1.0 + 0.0im, Nary_reverse(n))
    end
    function minus(t::Int)
      n = Nary_trans(t)
      n[(site-1)%site_number+1] -= 1
      return (1.0 + 0.0im, Nary_reverse(n))
    end
    return -0.5im * (f_op(plus) - f_op(minus))
  else
    throw(ArgumentError("kind must be '+', '-', 'x', 'y', or 'z'"))
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
function sum_j_fixed(f::Function, k::Int=1)
  sum_j(Tuple(f(i) for i in 1:(_site[]-k))...)
end
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
function true_f(i::Int, ma::Int)
  return true
end
function shift_z2(i::Int, ma::Int)
  if 2 * i == ma || i == ma
    return true
  else
    return false
  end
end
function block_generate(H_mat::Matrix{ComplexF64}, indices::Vector{Int})
  n = length(indices)
  H_mat1 = zeros(ComplexF64, n, n)
  for i in 1:n
    for j in 1:n
      H_mat1[i, j] = H_mat[indices[i], indices[j]]
    end
  end
  return H_mat1
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
      println(abs(imag(i)))
      throw(ArgumentError("The input vector contains complex numbers."))
    end
    push!(energy3, real(i))
  end
  return energy3
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
#region u1を分けたブロック対角化の途中
# function block_diag(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Int}, uf::UnionFinder{Int})
#   H_mat = mat_gene_nonsparse(H)
#   return eigen(H_mat)
# end
# function block_diag(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Int}, uf::UnionFinder{Int}, cases_ops_u::Tuple{Function,Op,Bool}...)
#   H_mat = mat_gene(H)
#   case_op_u1 = cases_ops_u[1]
#   case1 = case_op_u1[1]
#   op1 = case_op_u1[2]
#   u1 = case_op_u1[3]
#   if u1
#     eigens_dict = Dict{Float64,Vector{Int}}()
#     for i in indices
#       if i != (op1*i)[2]
#         throw(ArgumentError("The operator does not preserve the state."))
#       end
#       eigen1 = (op1*i)[1]
#       if haskey(eigens_dict, eigen1)
#         push!(eigens_dict[eigen1], i)
#       else
#         eigens_dict[eigen1] = [i]
#       end
#     end
#     eigens_sorted = sort(collect(keys(eigens_dict)))
#     energys1 = Vector{Float64}()
#     energyvectors1 = Vector{Vector{ComplexF64}}()
#     for eigen1 in eigens_sorted
#       H_mat1 = block_generate(H_mat, eigens_dict[eigen1])
#       eigens1, eigenvectors1 = block_diag(H_mat1, eigens_dict[eigen1], cases_ops_u[2:end]...)
#       energys1 = vcat_energy(energys1, eigens1)
#     end
#   else
#   end
# end
# function block_diag(H::Union{Op,Matrix{ComplexF64}}, cases_ops_u::Tuple{Function,Op,Bool}...)
#   return block_diag(H, collect(1:_dim[]^_site[]), _uf[], cases_ops_u...)
# end
# function block_diag(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Vector{Int}}, uf::UnionFinder{Int})
#   H_mat = mat_gene_nonsparse(H)
#   return eigen(H_mat)
# end
#endregion

function block_diag_energy(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Int}, uf::UnionFinder{Int}, cases_ops::Tuple{Function,Op}...)
  H_mat = mat_gene(H)
  case_op = cases_ops[1]
  case1 = case_op[1]
  op1 = case_op[2]
  # 最大の回数で戻ってくるものを探そう、あとop1をn回作用させたときの固有値ごとにVector{Vector{Int}}を作ろう、あと次のUnionFindを作ろう
  ma = 0
  index_set = Set{Int}()
  ntheigens_dict = Dict{Float64,Vector{Vector{Int}}}()
  uf1 = deepcopy(uf)
  for i in indices
    # すでにindex_setに入っているならcontinueしよう
    if i in index_set
      continue
    end
    # 忘れずにindex_setに追加しよう
    push!(index_set, i)
    # 代表元のVectorを作ろう
    rep_v = Vector{Int}()
    push!(rep_v, i)
    # unionfindの代表元になっているかを確認しよう
    if find!(uf, i) != i
      throw(ArgumentError("The index is not a representative of the union-find structure."))
    end
    # op1を適用して戻ってくるまでの回数を数えよう
    count = 0
    i1 = copy(i)
    ntheigen1 = 0.0
    while true
      count += 1
      # 演算子で作用させるときには0-indexに注意しよう
      val = (op1*(i1-1))[1]
      i1 = (op1*(i1-1))[2] + 1
      i2 = find!(uf, i1)
      if i2 == i
        ntheigen1 = val
        break
      end
      # ちゃんとunionfindの代表元をindex_setに追加しよう
      push!(index_set, i2)
      # 代表元を更新しよう
      union!(uf1, i, i2)
      # 代表元のVectorに追加しよう
      push!(rep_v, i2)
      if count > 1e10
        throw(ArgumentError("The operator does not preserve the state."))
      end
    end
    # 最大の回数を更新しよう
    ma = max(ma, count)
    # ntheigens_dictに追加しよう
    if haskey(ntheigens_dict, ntheigen1)
      push!(ntheigens_dict[ntheigen1], rep_v)
    else
      ntheigens_dict[ntheigen1] = [rep_v]
    end
  end
  # indicesの逆のDictを作成しよう
  indices_dict_inv = Dict{Int,Int}()
  for (i, index) in enumerate(indices)
    indices_dict_inv[index] = i
  end
  # 固有値ごとにブロック対角化した後のハミルトニアンとそのindexを作成しよう
  ntheigens_sorted = sort(collect(keys(ntheigens_dict)))
  nthroot1 = nthroots(1.0 + 0.0im, ma)
  energys1 = Vector{Float64}()
  energyvectors1 = Vector{Vector{ComplexF64}}()
  for ntheigen1 in ntheigens_sorted
    for i in 1:ma
      # ブロック対角化のためのインデックスを作成しよう
      indices1 = Vector{Int}()
      indices_dict1 = Dict{Int,Vector{Int}}()
      for v in ntheigens_dict[ntheigen1]
        if (length(v) * i) % ma != 0
          continue
        end
        push!(indices1, v[1])
        # v1 = Vector{Int}()
        # for j in v
        #   push!(v1,find!(uf,j))
        # end
        indices_dict1[v[1]] = v
      end
      println(indices1)
      # ブロック対角化のためのハミルトニアンを作成しよう
      H_mat1 = zeros(ComplexF64, length(indices1), length(indices1))
      for j in eachindex(indices1)
        for k in eachindex(indices1)
          coeff1 = sqrt(length(indices_dict1[indices1[k]])) / sqrt(length(indices_dict1[indices1[j]]))
          for l in eachindex(indices_dict1[indices1[j]])
            H_mat1[j, k] += coeff1 * H_mat[indices_dict_inv[indices_dict1[indices1[j]][l]], indices_dict_inv[indices1[k]]] * nthroot1[(i*(l-1)+ma-1)%ma+1]
          end
        end
      end
      # println(H_mat1)
      # ブロック対角化を行おう
      if length(cases_ops) == 1 || (!case1(i, ma))
        eigens1, eigenvectors1 = eigen(H_mat1)
      else
        # ここでcases_opsの2番目以降の要素を使ってブロック対角化を行う
        eigens1, eigenvectors1 = block_diag_energy(H_mat1, indices1, uf1, cases_ops[2:end]...)
      end
      # 固有値をエネルギーに変換しよう
      println(length(cases_ops))
      println(eigens1)
      energys1 = vcat_energy(energys1, eigens1)
    end
  end
  return (energys1, energyvectors1)
end
function block_diag_energy_shiftz2(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Vector{Int}}, uf::UnionFinder{Int}, k::Int, cases_ops::Tuple{Function,Op}...)
  H_mat = mat_gene(H)
  case_op = cases_ops[1]
  case1 = case_op[1]
  op1 = case_op[2]
  # 最大の回数で戻ってくるものを探そう、あとop1をn回作用させたときの固有値ごとにVector{Vector{Int}}を作ろう、あと次のUnionFindを作ろう
  ma = 0
  index_set = Set{Int}()
  ntheigens_dict = Dict{Float64,Vector{Vector{Int}}}()
  uf1 = deepcopy(uf)
  for v in indices
    i = v[1]
    # すでにindex_setに入っているならcontinueしよう
    if i in index_set
      continue
    end
    # 忘れずにindex_setに追加しよう
    push!(index_set, i)
    # 代表元のVectorを作ろう
    rep_v = Vector{Int}()
    push!(rep_v, i)
    # unionfindの代表元になっているかを確認しよう
    if find!(uf, i) != i
      throw(ArgumentError("The index is not a representative of the union-find structure."))
    end
    # op1を適用して戻ってくるまでの回数を数えよう
    count = 0
    i1 = copy(i)
    ntheigen1 = 0.0
    while true
      count += 1
      # 演算子で作用させるときには0-indexに注意しよう
      val = (op1*(i1-1))[1]
      i1 = (op1*(i1-1))[2] + 1
      i2 = find!(uf, i1)
      if i2 == i
        ntheigen1 = val
        break
      end
      # ちゃんとunionfindの代表元をindex_setに追加しよう
      push!(index_set, i2)
      # 代表元を更新しよう
      union!(uf1, i, i2)
      # 代表元のVectorに追加しよう
      push!(rep_v, i2)
      if count > 1e10
        throw(ArgumentError("The operator does not preserve the state."))
      end
    end
    # 最大の回数を更新しよう
    ma = max(ma, count)
    # ntheigens_dictに追加しよう
    if haskey(ntheigens_dict, ntheigen1)
      push!(ntheigens_dict[ntheigen1], rep_v)
    else
      ntheigens_dict[ntheigen1] = [rep_v]
    end
  end
  if ma == 2 && k != 0
    dict2 = Dict{Int,Tuple{Int,Int}}()
    for i in eachindex(indices)
      for j in eachindex(indices[i])
        dict2[indices[i][j]] = (i, j)
      end
    end
    v1 = Vector{Vector{Int}}()
    for i in eachindex(ntheigens_dict[1.0])
      if length(ntheigens_dict[1.0][i]) == 1 && dict2[(op1*(ntheigens_dict[1.0][i][1]-1))[2]+1][2] % 2 == 0
        push!(v1, deepcopy(ntheigens_dict[1.0][i]))
        ntheigens_dict[1.0][i][1] = -ntheigens_dict[1.0][i][1]
      elseif length(ntheigens_dict[1.0][i]) == 2 && dict2[(op1*(ntheigens_dict[1.0][i][1]-1))[2]+1][2] % 2 == 1
        push!(v1, ntheigens_dict[1.0][i])
      elseif length(ntheigens_dict[1.0][i]) == 2 && dict2[(op1*(ntheigens_dict[1.0][i][1]-1))[2]+1][2] % 2 == 0
        ntheigens_dict[1.0][i] = -ntheigens_dict[1.0][i]
        push!(v1, ntheigens_dict[1.0][i])
      end
    end
    ntheigens_dict[-1.0] = v1
    #println(ntheigens_dict)
  end
  # indicesの逆のDictを作成しよう
  indices_dict_inv = Dict{Int,Int}()
  for (i, index) in enumerate(indices)
    indices_dict_inv[index[1]] = i
  end
  # 固有値ごとにブロック対角化した後のハミルトニアンとそのindexを作成しよう
  ntheigens_sorted = sort(collect(keys(ntheigens_dict)))
  nthroot1 = nthroots(1.0 + 0.0im, ma)
  energys1 = Vector{Float64}()
  energyvectors1 = Vector{Vector{ComplexF64}}()
  for ntheigen1 in ntheigens_sorted
    for i in 1:ma
      if ma == 2 && k != 0 && ((ntheigen1 == 1.0 && i == 1) || (ntheigen1 == -1.0 && i == 2))
        continue
      end
      # ブロック対角化のためのインデックスを作成しよう
      indices1 = Vector{Vector{Int}}()
      indices_dict1 = Dict{Int,Vector{Int}}()
      for v in ntheigens_dict[ntheigen1]
        if v[1] < 0 && length(v) == 1
          continue
        end
        if (length(v) * i) % ma != 0 && (!(ma == 2 && k != 0 && ntheigen1 == -1.0 && i == 1))
          continue
        end
        push!(indices1, v)
        # v1 = Vector{Int}()
        # for j in v
        #   push!(v1,find!(uf,j))
        # end
        indices_dict1[v[1]] = v
      end
      # if ma == 2 && k !== 0
      #   println(ntheigen1)
      #   println(i)
      #   println(indices1)
      #   println(indices_dict1)
      # end
      #println(indices1)
      # ブロック対角化のためのハミルトニアンを作成しよう
      H_mat1 = zeros(ComplexF64, length(indices1), length(indices1))
      for j in eachindex(indices1)
        for m in eachindex(indices1)
          coeff1 = sqrt(length(indices_dict1[indices1[m][1]])) / sqrt(length(indices_dict1[indices1[j][1]]))
          for l in eachindex(indices_dict1[indices1[j][1]])
            # if ma == 2 && k != 0 && ntheigen1 == -1.0 && i == 1 && ((length(indices1[j]) == 1 && length(indices1[m]) == 2) || (length(indices1[j]) == 2 && length(indices1[m]) == 1))
            #   H_mat1[j, m] += coeff1 * H_mat[indices_dict_inv[indices_dict1[indices1[j][1]][l]], indices_dict_inv[indices1[m][1]]] * nthroot1[((i-1)*(l-1)+ma-1)%ma+1]
            #   continue
            # end
            if ma == 2 && k != 0 && ((ntheigen1 == -1.0 && i == 1) || (ntheigen1 == 1.0 && i == 2)) && ((indices1[j][1] < 0))
              H_mat1[j, m] += coeff1 * H_mat[indices_dict_inv[abs(indices_dict1[indices1[j][1]][l])], indices_dict_inv[abs(indices1[m][1])]] * nthroot1[((3-i)*(l-1)+ma-1)%ma+1]
              continue
            end
            H_mat1[j, m] += coeff1 * H_mat[indices_dict_inv[abs(indices_dict1[indices1[j][1]][l])], indices_dict_inv[abs(indices1[m][1])]] * nthroot1[(i*(l-1)+ma-1)%ma+1]
          end
        end
      end
      #println("H_mat1")
      # println(H_mat1)
      # ブロック対角化を行おう
      if length(cases_ops) == 1 || (!case1(i, ma))
        eigens1, eigenvectors1 = eigen(H_mat1)
        if length(eigens1) >= 3 && i == floor(_site[] / 2 - 0.1)
          println("ma=", ma, "i=", i, "ntheigen1=", ntheigen1)
          println(energy_gap(eigens1))
        end
      else
        # ここでcases_opsの2番目以降の要素を使ってブロック対角化を行う
        eigens1, eigenvectors1 = block_diag_energy_shiftz2(H_mat1, indices1, uf1, ma - i, cases_ops[2:end]...)
      end
      # 固有値をエネルギーに変換しよう
      #println(length(cases_ops))
      #println(eigens1)
      energys1 = vcat_energy(energys1, eigens1)
    end
  end
  return (energys1, energyvectors1)
end
function block_diag_energy_shiftz2(H::Union{Op,Matrix{ComplexF64}}, cases_ops::Tuple{Function,Op}...)
  uf = UnionFinder(_dim[]^_site[])
  return block_diag_energy_shiftz2(H, [[i] for i in 1:_dim[]^_site[]], uf, 0, cases_ops...)
end
function energy_same_check(energy1::Vector{Float64}, energy2::Vector{Float64})
  if length(energy1) != length(energy2)
    throw(ArgumentError("The lengths of the two energy vectors are not equal."))
  end
  sorted_energy1 = sort(energy1)
  sorted_energy2 = sort(energy2)
  b = true
  for i in 1:length(energy1)
    if abs(sorted_energy1[i] - sorted_energy2[i]) > 1.0e-10
      #println("i=", i)
      #println(sorted_energy1[i], sorted_energy2[i])
      #println(abs(sorted_energy1[i] - sorted_energy2[i]))
      b = false
    end
  end
  if !b
    #println(sorted_energy1)
    #println(sorted_energy2)
  end
  return b
end
function entanglment_entropy(energyvector::Vector{ComplexF64}, nsite::Int=floor(Int, _site[] / 2))
  if length(energyvector) < _dim[]^_site[]
    throw(ArgumentError("The energy vector must contain at least site elements."))
  end
  mat = zeros(ComplexF64, _dim[]^nsite, _dim[]^(_site[] - nsite))
  for i in 1:_dim[]^nsite
    for j in 1:_dim[]^(_site[]-nsite)
      mat[i, j] = energyvector[(i-1)*(_dim[]^(_site[]-nsite))+j]
    end
  end
  mat_S = svd(mat).S
  ans = 0.0
  for i in eachindex(mat_S)
    ans += -abs(mat_S[i])^2 * log(abs(mat_S[i])^2)
  end
  return ans
end
function entanglement_entropy_show(energy::Vector{Float64}, energyvector::Matrix{ComplexF64}, nsite::Int=floor(Int, _site[] / 2))
  vector1 = Vector{Float64}(undef, length(energy))
  for i in eachindex(energy)
    energyvector1 = energyvector[:, i]
    entanglment_entropy1 = entanglment_entropy(energyvector1, nsite)
    vector1[i] = entanglment_entropy1
  end
  return (energy, vector1)
end
function block_diag_entropy(H::Union{Op,Matrix{ComplexF64}}, cases_ops::Tuple{Function,Op}...)
  uf1 = UnionFinder(_dim[]^_site[])
  return block_diag_entropy(H, collect(1:_dim[]^_site[]), uf1, cases_ops...)
end
function block_diag_entropy(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Int}, uf::UnionFinder{Int}, cases_ops::Tuple{Function,Op}...)
  H_mat = mat_gene(H)
  case_op = cases_ops[1]
  case1 = case_op[1]
  op1 = case_op[2]
  # 最大の回数で戻ってくるものを探そう、あとop1をn回作用させたときの固有値ごとにVector{Vector{Int}}を作ろう、あと次のUnionFindを作ろう
  ma = 0
  index_set = Set{Int}()
  ntheigens_dict = Dict{Float64,Vector{Vector{Int}}}()
  uf1 = deepcopy(uf)
  for i in indices
    # すでにindex_setに入っているならcontinueしよう
    if i in index_set
      continue
    end
    # 忘れずにindex_setに追加しよう
    push!(index_set, i)
    # 代表元のVectorを作ろう
    rep_v = Vector{Int}()
    push!(rep_v, i)
    # unionfindの代表元になっているかを確認しよう
    if find!(uf, i) != i
      throw(ArgumentError("The index is not a representative of the union-find structure."))
    end
    # op1を適用して戻ってくるまでの回数を数えよう
    count = 0
    i1 = copy(i)
    ntheigen1 = 0.0
    while true
      count += 1
      # 演算子で作用させるときには0-indexに注意しよう
      val = (op1*(i1-1))[1]
      i1 = (op1*(i1-1))[2] + 1
      i2 = find!(uf, i1)
      if i2 == i
        ntheigen1 = val
        break
      end
      # ちゃんとunionfindの代表元をindex_setに追加しよう
      push!(index_set, i2)
      # 代表元を更新しよう
      union!(uf1, i, i2)
      # 代表元のVectorに追加しよう
      push!(rep_v, i2)
      if count > 1e10
        throw(ArgumentError("The operator does not preserve the state."))
      end
    end
    # 最大の回数を更新しよう
    ma = max(ma, count)
    # ntheigens_dictに追加しよう
    if haskey(ntheigens_dict, ntheigen1)
      push!(ntheigens_dict[ntheigen1], rep_v)
    else
      ntheigens_dict[ntheigen1] = [rep_v]
    end
  end
  # indicesの逆のDictを作成しよう
  indices_dict_inv = Dict{Int,Int}()
  for (i, index) in enumerate(indices)
    indices_dict_inv[index] = i
  end
  # 固有値ごとにブロック対角化した後のハミルトニアンとそのindexを作成しよう
  ntheigens_sorted = sort(collect(keys(ntheigens_dict)))
  nthroot1 = nthroots(1.0 + 0.0im, ma)
  energys1 = Vector{Float64}()
  energyvectors1 = Vector{Vector{ComplexF64}}()
  for ntheigen1 in ntheigens_sorted
    for i in 1:ma
      # ブロック対角化のためのインデックスを作成しよう
      indices1 = Vector{Int}()
      indices_dict1 = Dict{Int,Vector{Int}}()
      for v in ntheigens_dict[ntheigen1]
        if (length(v) * i) % ma != 0
          continue
        end
        push!(indices1, v[1])
        # v1 = Vector{Int}()
        # for j in v
        #   push!(v1,find!(uf,j))
        # end
        indices_dict1[v[1]] = v
      end
      # ブロック対角化のためのハミルトニアンを作成しよう
      H_mat1 = zeros(ComplexF64, length(indices1), length(indices1))
      for j in eachindex(indices1)
        for k in eachindex(indices1)
          coeff1 = sqrt(length(indices_dict1[indices1[k]])) / sqrt(length(indices_dict1[indices1[j]]))
          for l in eachindex(indices_dict1[indices1[j]])
            H_mat1[j, k] += coeff1 * H_mat[indices_dict_inv[indices_dict1[indices1[j]][l]], indices_dict_inv[indices1[k]]] * nthroot1[(i*(l-1)+ma-1)%ma+1]
          end
        end
      end
      # println(H_mat1)
      # ブロック対角化を行おう
      if length(cases_ops) == 1 || (!case1(i, ma))
        eigens1, eigenvectors1 = eigen(H_mat1)
        mat1 = zeros(ComplexF64, _dim[]^_site[], length(eigens1))
        # println(length(eigens1))
        # println(ntheigens_dict[ntheigen1][i])
        for j in eachindex(eigens1)
          for k in eachindex(eigenvectors1[:, j])
            mat1[ntheigens_dict[ntheigen1][k][1], j] = eigenvectors1[k, j]
          end
        end
        if ntheigen1 == 0.0
          return (eigens1, mat1)
        end
      else
        # ここでcases_opsの2番目以降の要素を使ってブロック対角化を行う
        eigens1, eigenvectors1 = block_diag_energy(H_mat1, indices1, uf1, cases_ops[2:end]...)
      end
      # 固有値をエネルギーに変換しよう
      energys1 = vcat_energy(energys1, eigens1)
    end
  end
  return (energys1, energyvectors1)
end
function block_diag_entropy_z2(H::Union{Op,Matrix{ComplexF64}}, op1::Op)
  uf1 = UnionFinder(_dim[]^_site[])
  return block_diag_entropy_z2(H, collect(1:_dim[]^_site[]), uf1, op1)
end
function block_diag_entropy_z2(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Int}, uf::UnionFinder{Int}, op1::Op)
  H_mat = mat_gene(H)
  # 最大の回数で戻ってくるものを探そう、あとop1をn回作用させたときの固有値ごとにVector{Vector{Int}}を作ろう、あと次のUnionFindを作ろう
  ma = 0
  index_set = Set{Int}()
  ntheigens_dict = Dict{Float64,Vector{Vector{Int}}}()
  uf1 = deepcopy(uf)
  for i in indices
    # すでにindex_setに入っているならcontinueしよう
    if i in index_set
      continue
    end
    # 忘れずにindex_setに追加しよう
    push!(index_set, i)
    # 代表元のVectorを作ろう
    rep_v = Vector{Int}()
    push!(rep_v, i)
    # unionfindの代表元になっているかを確認しよう
    if find!(uf, i) != i
      throw(ArgumentError("The index is not a representative of the union-find structure."))
    end
    # op1を適用して戻ってくるまでの回数を数えよう
    count = 0
    i1 = copy(i)
    ntheigen1 = 0.0
    while true
      count += 1
      # 演算子で作用させるときには0-indexに注意しよう
      val = (op1*(i1-1))[1]
      i1 = (op1*(i1-1))[2] + 1
      i2 = find!(uf, i1)
      if i2 == i
        ntheigen1 = val
        break
      end
      # ちゃんとunionfindの代表元をindex_setに追加しよう
      push!(index_set, i2)
      # 代表元を更新しよう
      union!(uf1, i, i2)
      # 代表元のVectorに追加しよう
      push!(rep_v, i2)
      if count > 1e10
        throw(ArgumentError("The operator does not preserve the state."))
      end
    end
    # 最大の回数を更新しよう
    ma = max(ma, count)
    # ntheigens_dictに追加しよう
    if haskey(ntheigens_dict, ntheigen1)
      push!(ntheigens_dict[ntheigen1], rep_v)
    else
      ntheigens_dict[ntheigen1] = [rep_v]
    end
  end
  if ma != 2
    throw(ArgumentError("The maximum number of iterations must be 2."))
  end
  # indicesの逆のDictを作成しよう
  indices_dict_inv = Dict{Int,Int}()
  for (i, index) in enumerate(indices)
    indices_dict_inv[index] = i
  end
  # 固有値ごとにブロック対角化した後のハミルトニアンとそのindexを作成しよう
  ntheigens_sorted = sort(collect(keys(ntheigens_dict)))
  nthroot1 = nthroots(1.0 + 0.0im, ma)
  energys1 = Vector{Float64}()
  energyvectors1 = zeros(ComplexF64, _dim[]^_site[], 0)
  for ntheigen1 in ntheigens_sorted
    for i in 1:ma
      # ブロック対角化のためのインデックスを作成しよう
      indices1 = Vector{Int}()
      indices_dict1 = Dict{Int,Vector{Int}}()
      for v in ntheigens_dict[ntheigen1]
        if (length(v) * i) % ma != 0
          continue
        end
        push!(indices1, v[1])
        # v1 = Vector{Int}()
        # for j in v
        #   push!(v1,find!(uf,j))
        # end
        indices_dict1[v[1]] = v
      end
      # ブロック対角化のためのハミルトニアンを作成しよう
      H_mat1 = zeros(ComplexF64, length(indices1), length(indices1))
      for j in eachindex(indices1)
        for k in eachindex(indices1)
          coeff1 = sqrt(length(indices_dict1[indices1[k]])) / sqrt(length(indices_dict1[indices1[j]]))
          for l in eachindex(indices_dict1[indices1[j]])
            H_mat1[j, k] += coeff1 * H_mat[indices_dict_inv[indices_dict1[indices1[j]][l]], indices_dict_inv[indices1[k]]] * nthroot1[(i*(l-1)+ma-1)%ma+1]
          end
        end
      end
      # println(H_mat1)
      # ブロック対角化を行おう
      eigens1, eigenvectors1 = eigen(H_mat1)
      mat1 = zeros(ComplexF64, _dim[]^_site[], length(eigens1))
      # eigenvectorsを普通の基底に戻そう
      for j in eachindex(eigens1)
        for k in eachindex(eigenvectors1[:, j])
          for l in eachindex(ntheigens_dict[ntheigen1][k])
            coeff1 = 0.0 + 0.0im
            if length(ntheigens_dict[ntheigen1][k]) == 2 && i == ma
              coeff1 = 0.5 + 0.0im
            else
              coeff1 = 1.0 / sqrt(length(ntheigens_dict[ntheigen1][k])) + 0.0im
            end
            mat1[ntheigens_dict[ntheigen1][k][l], j] += eigenvectors1[k, j] * nthroot1[mod((-i * (l - 1) + ma - 1), ma)+1] * coeff1
          end
        end
      end
      if i == 2
        eigens2 = vcat_energy(Float64[], eigens1)
        return entanglement_entropy_show(eigens2, mat1)
      end
      energyvectors1 = hcat(energyvectors1, mat1)
      # 固有値をエネルギーに変換しよう
      energys1 = vcat_energy(energys1, eigens1)
    end
  end
  return (energys1, transpose(energyvectors1))
end
function block_diag_entropy_u1z2(H::Union{Op,Matrix{ComplexF64}}, cases_ops::Tuple{Function,Op}...)
  uf1 = UnionFinder(_dim[]^_site[])
  return block_diag_entropy(H, collect(1:_dim[]^_site[]), uf1, cases_ops...)
end
function block_diag_entropy_u1z2(H::Union{Op,Matrix{ComplexF64}}, indices::Vector{Int}, uf::UnionFinder{Int}, cases_ops::Tuple{Function,Op}...)
  H_mat = mat_gene(H)
  case_op = cases_ops[1]
  case1 = case_op[1]
  op1 = case_op[2]
  # 最大の回数で戻ってくるものを探そう、あとop1をn回作用させたときの固有値ごとにVector{Vector{Int}}を作ろう、あと次のUnionFindを作ろう
  ma = 0
  index_set = Set{Int}()
  ntheigens_dict = Dict{Float64,Vector{Vector{Int}}}()
  uf1 = deepcopy(uf)
  for i in indices
    # すでにindex_setに入っているならcontinueしよう
    if i in index_set
      continue
    end
    # 忘れずにindex_setに追加しよう
    push!(index_set, i)
    # 代表元のVectorを作ろう
    rep_v = Vector{Int}()
    push!(rep_v, i)
    # unionfindの代表元になっているかを確認しよう
    if find!(uf, i) != i
      throw(ArgumentError("The index is not a representative of the union-find structure."))
    end
    # op1を適用して戻ってくるまでの回数を数えよう
    count = 0
    i1 = copy(i)
    ntheigen1 = 0.0
    while true
      count += 1
      # 演算子で作用させるときには0-indexに注意しよう
      val = (op1*(i1-1))[1]
      i1 = (op1*(i1-1))[2] + 1
      i2 = find!(uf, i1)
      if i2 == i
        ntheigen1 = val
        break
      end
      # ちゃんとunionfindの代表元をindex_setに追加しよう
      push!(index_set, i2)
      # 代表元を更新しよう
      union!(uf1, i, i2)
      # 代表元のVectorに追加しよう
      push!(rep_v, i2)
      if count > 1e10
        throw(ArgumentError("The operator does not preserve the state."))
      end
    end
    # 最大の回数を更新しよう
    ma = max(ma, count)
    # ntheigens_dictに追加しよう
    if haskey(ntheigens_dict, ntheigen1)
      push!(ntheigens_dict[ntheigen1], rep_v)
    else
      ntheigens_dict[ntheigen1] = [rep_v]
    end
  end
  # indicesの逆のDictを作成しよう
  indices_dict_inv = Dict{Int,Int}()
  for (i, index) in enumerate(indices)
    indices_dict_inv[index] = i
  end
  # 固有値ごとにブロック対角化した後のハミルトニアンとそのindexを作成しよう
  ntheigens_sorted = sort(collect(keys(ntheigens_dict)))
  nthroot1 = nthroots(1.0 + 0.0im, ma)
  energys1 = Vector{Float64}()
  energyvectors1 = Vector{Vector{ComplexF64}}()
  for ntheigen1 in ntheigens_sorted
    for i in 1:ma
      # ブロック対角化のためのインデックスを作成しよう
      indices1 = Vector{Int}()
      indices_dict1 = Dict{Int,Vector{Int}}()
      for v in ntheigens_dict[ntheigen1]
        if (length(v) * i) % ma != 0
          continue
        end
        push!(indices1, v[1])
        # v1 = Vector{Int}()
        # for j in v
        #   push!(v1,find!(uf,j))
        # end
        indices_dict1[v[1]] = v
      end
      # ブロック対角化のためのハミルトニアンを作成しよう
      H_mat1 = zeros(ComplexF64, length(indices1), length(indices1))
      for j in eachindex(indices1)
        for k in eachindex(indices1)
          coeff1 = sqrt(length(indices_dict1[indices1[k]])) / sqrt(length(indices_dict1[indices1[j]]))
          for l in eachindex(indices_dict1[indices1[j]])
            H_mat1[j, k] += coeff1 * H_mat[indices_dict_inv[indices_dict1[indices1[j]][l]], indices_dict_inv[indices1[k]]] * nthroot1[(i*(l-1)+ma-1)%ma+1]
          end
        end
      end
      # println(H_mat1)
      # ブロック対角化を行おう
      if length(cases_ops) == 1 || (!case1(i, ma))
        println("Block diagonalization for case 1")
        eigens1, eigenvectors1 = eigen(H_mat1)
        mat1 = zeros(ComplexF64, _dim[]^_site[], length(eigens1))
        for j in eachindex(eigens1)
          for k in eachindex(eigenvectors1[:, j])
            mat1[ntheigens_dict[ntheigen1][k][1], j] = eigenvectors1[k, j]
          end
        end
        for l in eachindex(ntheigens_dict[ntheigen1][k])
          coeff1 = 0.0 + 0.0im
          if length(ntheigens_dict[ntheigen1][k]) == 2 && i == ma
            coeff1 = 0.5 + 0.0im
          else
            coeff1 = 1.0 / sqrt(length(ntheigens_dict[ntheigen1][k])) + 0.0im
          end
          mat1[ntheigens_dict[ntheigen1][k][l], j] += eigenvectors1[k, j] * nthroot1[(i*(l-1)+ma-1)%ma+1] * coeff1
        end
        if length(indices) == binomial(_site[], floor(Int, _site[] / 2.0)) && i == 2
          println(ntheigens_dict[ntheigen1])
          return (eigens1, mat1)
        end
      else
        # ここでcases_opsの2番目以降の要素を使ってブロック対角化を行う
        eigens1, eigenvectors1 = block_diag_energy(H_mat1, indices1, uf1, cases_ops[2:end]...)
      end
      # 固有値をエネルギーに変換しよう
      energys1 = vcat_energy(energys1, eigens1)
    end
  end
  return (energys1, energyvectors1)
end
end
using .ExactDiag
using UnionFind
using LinearAlgebra
using Plots
default(dpi=600)
L = 2
hj = 1.0
# hj=sqrt(10000.0)
vj = 1.0
kj = sqrt(2.5)
nn = sqrt(0.8)
Δ = sqrt(1.3)
init(2, L)
# 横磁場縦磁場イジングモデル(開放端条件)--サイト反転対称性のみ
H1 = sum_j_fixed(j -> spin_op('z', j) * spin_op('z', j + 1)) - sum_j(j -> 0.5 * hj * (spin_op('+', j) + spin_op('-', j))) - vj * sum_j(j -> spin_op('z', j))
# 横磁場イジングモデル(開放端条件と端の改変)--スピン反転対称性のみ
#H2 = sum_j_fixed(j -> spin_op('z', j) * spin_op('z', j + 1), 2) + kj * spin_op('z', L - 1) * spin_op('z', L) - sum_j(j -> hj * (spin_op('+', j) + spin_op('-', j))) - vj * sum_j(j -> spin_op('z', j))
# 横磁場縦磁場イジングモデル(開放端条件と次近接)--非可積分でサイト反転対称性のみ
#H3 = sum_j_fixed(j -> 0.5 * spin_op('z', j) * spin_op('z', j + 1)) + 0.5 * nn * sum_j_fixed(j -> 0.5 * spin_op('z', j) * spin_op('z', j + 2), 2) - sum_j(j -> 0.5 * hj * (spin_op('+', j) + spin_op('-', j))) - vj * sum_j(j -> spin_op('z', j))
# XXZモデル(開放端条件)--個数保存対称性のみ
#H4 = sum_j_fixed(j -> 0.5 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)) + Δ * spin_op('z', j) * spin_op('z', j + 1)) + sum_j(j -> (vj * sin(j)) * spin_op('z', j))
#H5 = -1.0 / 2.0 * sum_j(j -> -0.5 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)))
#H6 = sum_j_fixed(j -> 0.25 * (spin_op('+', j) * spin_op('-', j + 1) + spin_op('-', j) * spin_op('+', j + 1)) + 0.5 * Δ * spin_op('z', j) * spin_op('z', j + 1)) + sum_j_fixed(j -> 0.25 * (spin_op('+', j) * spin_op('-', j + 2) + spin_op('-', j) * spin_op('+', j + 2)) + 0.5 * Δ * spin_op('z', j) * spin_op('z', j + 2), 2)
S_z = sum_j(j -> spin_op('z', j))
# println(block_diag_entropy_u1z2(H1, (true_f, S_z), (true_f, site_flip())))
x, y = block_diag_entropy_z2(H1, site_flip())
# eigens1, mat1 = block_diag_entropy_z2(H1, (true_f, site_flip()))
# x, y = entanglement_entropy_show(eigens1, mat1)
println(x, y)
# scatter(x, y;
#   label="eigenvalue=1",
#   xlabel="energy",
#   ylabel="entanglement entropy",
#   title="transverse field Ising model",
#   markersize=1.5, markerstrokewidth=0.5)
# cd(raw"\\wsl.localhost\Ubuntu\home\kokor\git\exact_diag")
# savefig("entropy_spinflip2.png")
