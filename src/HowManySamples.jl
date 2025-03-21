module HowManySamples

export mu_true, sigma_true, traceplot, Observations, Samples, exact_model, mc_model, expand_samples

using CairoMakie
using Distributions
using MCMCChains
using PairPlots
using StatsFuns
using Random
using Turing

const mu_true = 0.0
const sigma_true = 1.0

struct Observations
    x::Vector{Float64}
    xo::Vector{Float64}
    so::Vector{Float64}
end

struct Samples
    obs::Observations
    xs::Vector{Vector{Float64}}
    logwts::Vector{Vector{Float64}}
end

function Random.rand(::Type{Observations}, N::T) where T <: Integer
    x = rand(Normal(mu_true, sigma_true), N)
    so = rand(Uniform(sigma_true, 2*sigma_true), N)
    xo = [rand(Normal(x[i], so[i])) for i in 1:N]
    return Observations(x, xo, so)
end

function Random.rand(s::Type{Samples}, obs::Observations, Nsamp::T) where T <: Integer
    rand(s, obs, Nsamp*ones(Int, length(obs.xo)))
end
function Random.rand(::Type{Samples}, obs::Observations, Nsamp::Vector{<:Integer})
    prior_pdf = Normal(mu_true, 2*sigma_true) # Want a bit less shrinkage, but toward the right mean.
    result = map(obs.xo, obs.so, Nsamp) do x, s, N
        sigma = 1/sqrt(1/s^2 + 1/(2*sigma_true)^2)
        mu = ((2*sigma_true)^2 * x + s^2 * mu_true) / (s^2 + (2*sigma_true)^2)
        samples = rand(Normal(mu, sigma), N)
        logwts = [logpdf(prior_pdf, x) for x in samples]

        (samples, logwts)
    end
    
    Samples(obs, [x[1] for x in result], [x[2] for x in result])
end

@model function exact_model(obs)
    mu ~ Normal(0,1)
    sigma ~ Exponential(1)

    obs.xo ~ arraydist([Normal(mu, sqrt(sigma*sigma + s*s)) for s in obs.so])
end

@model function mc_model(samps)
    mu ~ Normal(0,1)
    sigma ~ Exponential(1)

    logps = map(samps.xs, samps.logwts) do samps, lwts
        map(samps, lwts) do x, lw
            logpdf(Normal(mu, sigma), x) - lw
        end
    end

    Turing.@addlogprob!(sum(map(logsumexp, logps)))

    lp_vars = map(logps) do lp
        mu = logsumexp(lp)
        p_scaled = exp.(lp .- mu)
        p_var = var(p_scaled)
        length(lp) * p_var
    end

    neff_samps = map(lp_vars) do lpv
        1/lpv
    end

    return (; neff_samps = neff_samps, lp_vars = lp_vars, total_lp_var = sum(lp_vars))
end

function traceplot(chns)
    params = names(chns, :parameters)

    n_chains = length(chains(chns))
    n_samples = length(chns)

    fig = Figure()
    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 1]; ylabel=string(param))
        for chain in 1:n_chains
            values = chns[:, param, chain]
            lines!(ax, 1:n_samples, values; label=string(chain))
        end
    
        hideydecorations!(ax; label=false)
        if i < length(params)
            hidexdecorations!(ax; grid=false)
        else
            ax.xlabel = "Iteration"
        end
    end

    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 2]; ylabel=string(param))
        for chain in 1:n_chains
            values = chns[:, param, chain]
            density!(ax, values; label=string(chain))
        end
    
        hideydecorations!(ax)
        if i < length(params)
            hidexdecorations!(ax; grid=false)
        else
            ax.xlabel = "Parameter estimate"
        end
    end
    
    axes = [only(contents(fig[i, 2])) for i in 1:length(params)]
    linkxaxes!(axes...)
    axislegend(first(axes))

    fig
end

end # module HowManySamples
