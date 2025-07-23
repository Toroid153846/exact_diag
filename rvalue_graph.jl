using Plots
x = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
y = [0.393716461236689,0.522511802228144,0.3196434924732828,0.47418203466915926,0.511009378326365,0.5157899414061143,0.5131539470912587,0.5194272915637081,0.5361713599081623,0.5221591589727458,0.530225251960061,0.5281407395179368]
plot(x, y, label="Longitudinal magnetic field ising model", xlabel="site number", ylabel="rvalue", title="rvalue", legend=:topright, marker=:circle, markersize=5, linewidth=2, color=:blue,ylims=(0.30, 0.60))
threshold = 0.5307
hline!([threshold];
  label="0.5307",
  linestyle=:dash, linewidth=2, color=:red)
# plot(x, y, label="Transverse magnetic field ising model", xlabel="site number", ylabel="rvalue", title="rvalue", legend=:topright, marker=:circle, markersize=5, linewidth=2, color=:blue)
# threshold = 2*log(2)-1
# hline!([threshold];
#   label="2*ln(2)-1",
#   linestyle=:dash, linewidth=2, color=:red)