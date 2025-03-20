using CairoMakie
using DataFrames
using HDF5
using HowManySamples
using LaTeXStrings
using MCMCChains
using MCMCChainsStorage
using PairPlots

include("paths.jl")

trace_exact, trace_mc = h5open(joinpath(data, "traces.h5"), "r") do file
    read(file["exact"], Chains), read(file["mc"], Chains)
end

p = pairplot(
    PairPlots.Series(
        trace_exact[[:mu, :sigma]],
        label = "Exact",
        color = Makie.wong_colors()[1],
        strokecolor = Makie.wong_colors()[1]
    ),
    PairPlots.Series(
        trace_mc[[:mu, :sigma]],
        label = "Monte-Carlo",
        color = Makie.wong_colors()[2],
        strokecolor = Makie.wong_colors()[2]
    ),
    PairPlots.Truth(
        Dict(
            :mu => mu_true,
            :sigma => sigma_true
        ),
        label="Truth",
        color=:black,
        strokecolor=:black
    ),
    labels = Dict(
        :mu => L"\mu",
        :sigma => L"\sigma"
    )
)

save(joinpath(figures, "mu_sigma_pairplot.pdf"), p)