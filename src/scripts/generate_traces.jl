using Distributed
addprocs(4)

@everywhere begin
    using HowManySamples
    using HDF5
    using MCMCChainsStorage
    using Random
    using Turing
end

function sample_mc(full_samp, Nsamp)
    samp = Samples(full_samp.obs, [x[1:Nsamp[i]] for (i, x) in enumerate(full_samp.xs)], [x[1:Nsamp[i]] for (i, x) in enumerate(full_samp.logwts)])
    mod = mc_model(samp)
    mc_trace = sample(mod, NUTS(1000, 0.85), MCMCDistributed(), 1000, 4)
    genq = generated_quantities(mod, mc_trace)
    mc_trace = append_generated_quantities(mc_trace, genq)
    return mc_trace
end

include("paths.jl")

Nobs = 128
Neff_thresh = 10

Random.seed!(0xe3e4085fd0b2c6c1)
obs = rand(Observations, Nobs)

exact_trace = sample(exact_model(obs), NUTS(1000, 0.85), MCMCDistributed(), 1000, 4)

Random.seed!(0xc506bfb940402ad9)
full_samp = rand(Samples, obs, 32768)
Nsamp = 16*ones(Int, length(obs.xo))
mc_trace = sample_mc(full_samp, Nsamp)

while true
    ns = [minimum(mc_trace["neff_samps[$(i)]"]) for i in 1:length(namesingroup(mc_trace, :neff_samps))]
    min_neff = minimum(ns)

    @info "Found min_neff = $(round(min_neff, digits=3))"

    if min_neff > Neff_thresh
        break
    end

    sel = ns .< Neff_thresh
    Nsamp[sel] .= 2*Nsamp[sel]

    mc_trace = sample_mc(full_samp, Nsamp)
end

h5open(joinpath(data, "traces.h5"), "w") do file
    exact_group = create_group(file, "exact")
    mc_group = create_group(file, "mc")

    write(exact_group, exact_trace)
    write(mc_group, mc_trace)
end