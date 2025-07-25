function eigens_dict(op1::Op)
  dim_tot = _dim[]^_site[]
  dict1 = Dict{Float64,Vector{Vector{Int}}}()
  set1 = Set{Int}()
  maximum = 0
  for i in 1:dim_tot
    if i in set1
      continue
    end
    i1 = copy(i)
    vec1 = Vector{Int}()
    push!(vec1, i1)
    push!(set1, i1)
    ntheigenvalue = 1.0
    for j in 1:dim_tot
      tup1 = op1 * i1
      ntheigenvalue *= tup1[1]
      if tup1[2] == i
        maximum = max(maximum, j)
        break
      end
      i1 = copy(tup1[2])
      push!(vec1, i1)
      push!(set1, i1)
    end
    println(ntheigenvalue)
    dict1[ntheigenvalue] = get(dict1, ntheigenvalue, Vector{Vector{Int}}())
    push!(dict1[ntheigenvalue], vec1)
  end
  return (maximum, dict1)
end
function nthroots(z::ComplexF64, n::Int)
  if n != 1 && abs(imag(z)) < 1.0e-10 && real(z) < 1.0e-10
    throw(ArgumentError("n must be a positive integer greater than 0."))
  end
  if n == 2 && abs(imag(z)) < 1.0e-10 && real(z) > 1.0e-10
    return [-sqrt(real(z)) + 0.0im, sqrt(real(z)) + 0.0im]
  end
  if n == 2 && abs(imag(z)) < 1.0e-10 && real(z) < -1.0e-10
    return [0.0 - (sqrt(real(z)))im, 0.0 + (sqrt(real(z)))im]
  end
  if n == 4 && abs(imag(z)) < 1.0e-10 && real(z) > 1.0e-10
    return [0.0 + ((real(z))^(1.0 / 4.0))im, -real(z)^(1.0 / 4.0) + 0.0im, 0.0 - ((real(z))^(1.0 / 4.0))im, (real(z))^(1.0 / 4.0) + 0.0im]
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