using HDF5
using MCMCChains
using MCMCChainsStorage

include("paths.jl")

result_mc = h5open(joinpath(data, "traces.h5"), "r") do file
    read(file["mc"], Chains)
end

max_lp_var = maximum(result_mc[:total_lp_var])
min_neff = minimum([minimum(result_mc[k]) for k in namesingroup(result_mc, :neff_samps)])

open(joinpath(output, "min_neff.txt"), "w") do io
    write(io, "$(round(min_neff, digits=1))")
end
open(joinpath(output, "max_lp_var.txt"), "w") do io
    write(io, "$(round(max_lp_var, digits=1))")
end