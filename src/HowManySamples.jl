module HowManySamples

export mu_true, sigma_true, traceplot, pairplot_results, run_one_simulation

using CairoMakie
using Distributions
using MCMCChains
using PairPlots
using StatsFuns
using Turing

const mu_true = 0.0
const sigma_true = 1.0

function draw_truth(N)
    return rand(Normal(mu_true, sigma_true), N)
end

function draw_obs_sigmas(N)
    return rand(Uniform(sigma_true, 2*sigma_true), N)
end

function draw_obs(x_true, sigma_obs)
    return rand.(Normal.(x_true, sigma_obs))
end

function draw_samples(xo, so, Nsamp::Int)
    return draw_samples(xo, so, Nsamp*ones(Int, length(xo)))
end

function draw_samples(xo, so, Nsamp)
    return [rand(Normal(x, s), N) for (x,s,N) in zip(xo, so, Nsamp)]
end

function draw_samples_and_weights(xo, so, Nsamp::Int)
    return draw_samples_and_weights(xo, so, Nsamp*ones(Int, length(xo)))
end
function draw_samples_and_weights(xo, so, Nsamp)
    true_normal = Normal(mu_true, sigma_true)
    samples_and_logweights = map(xo, so, Nsamp) do x, s, N
        s_total = sqrt(1/(1/s^2 + 1/sigma_true^2))
        m_total = (x/s^2 + mu_true/sigma_true^2) * s_total^2

        samples = rand(Normal(m_total, s_total), N)
        logwts = logpdf.((true_normal,), samples)

        (samples, logwts)
    end

    return [x[1] for x in samples_and_logweights], [x[2] for x in samples_and_logweights]
end

@model function exact_model(xo, so)
    mu ~ Normal(0,1)
    sigma ~ Exponential(1)

    xo ~ arraydist([Normal(mu, sqrt(sigma*sigma + s*s)) for s in so])
end

@model function mc_model(xsamples, log_wts)
    mu ~ Normal(0,1)
    sigma ~ Exponential(1)

    logps = map(xsamples, log_wts) do samps, lwts
        lps = map(samps, lwts) do x, lw
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

    neff_samps = map(lp_vars) do lp
        1/lp
    end

    return (; neff_samps = neff_samps, lp_vars = lp_vars, total_lp_var = sum(lp_vars))
end

function effective_sigma(sigma_obs)
    return sqrt(1 / sum( 1 ./ (sigma_obs .* sigma_obs)))
end

function run_exact_simulation(N)
    xt = draw_truth(N)
    so = draw_obs_sigmas(N)
    xo = draw_obs(xt, so)
    model = exact_model(xo, so)
    trace = sample(model, NUTS(1000, 0.85), 1000)
    return (; trace = trace, x_true = xt, x_obs = xo, sigma_obs = so)
end

function run_sample_simulation(xs, logwts)
    model = mc_model(xs, logwts)
    trace = sample(model, NUTS(1000, 0.85), 1000)
    genq = generated_quantities(model, trace)
    return (; trace = trace, genq = genq)
end

function min_neff_samps(genq)
    return minimum([minimum(x.neff_samps) for x in genq])
end

function max_lp_var(genq)
    return maximum([maximum(x.lp_vars) for x in genq])
end

function update_samples(xo, xs, xsamps, logwts, genq, min_neff)
    neff = [minimum([g.neff_samps[i] for g in genq]) for i in eachindex(xsamps)]
    nsamp = [length(x) for x in xsamps]

    ndraw = [(ne > min_neff ? 0 : ns) for (ne, ns) in zip(neff, nsamp)]

    new_xsamps, new_logwts = draw_samples_and_weights(xo, xs, ndraw)

    return [vcat(x, nx) for (x, nx) in zip(xsamps, new_xsamps)], [vcat(lw, nlw) for (lw, nlw) in zip(logwts, new_logwts)]
end

function correct_nsamp(N, ne, desired_ne)
    if ne > 2*desired_ne
        Int(round(N/2))
    elseif ne < desired_ne/2
        Int(round(2*N))
    else
        Int(round(N))
    end
end

function update_nsamp(genq, Nsamp::Int, desired_neff)
    update_nsamp(genq, Nsamp*ones(Int, length(first(genq).neff_samps)), desired_neff)
end

function update_nsamp(genq, Nsamp, desired_neff)
    min_neffs = [minimum(vec([x.neff_samps[i] for x in genq])) for i in eachindex(first(genq).neff_samps)]
    return [correct_nsamp(Nsamp[i], min_neffs[i], desired_neff) for i in eachindex(Nsamp)]
end

function pairplot_results(result, result_mc)
    pairplot(PairPlots.Series(result.trace[[:mu, :sigma]], label="Exact", color=Makie.wong_colors()[1], strokecolor=Makie.wong_colors()[1]), PairPlots.Series(result_mc.trace[[:mu, :sigma]], label="Monte-Carlo", color=Makie.wong_colors()[2], strokecolor=Makie.wong_colors()[2]), PairPlots.Truth((; mu=mu_true, sigma=sigma_true), color=:black, label="Truth"))
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


function run_one_simulation(; N = 128, Nsamp_init = 16, Neff_min = 10)
    Nsamp = Nsamp_init

    result_exact = run_exact_simulation(N)

    xsamps, logwts = draw_samples_and_weights(result_exact.x_obs, result_exact.sigma_obs, Nsamp)

    result_mc = run_sample_simulation(xsamps, logwts)
    @info "Found $(round(min_neff_samps(result_mc.genq), digits=2)) effective samples, digits=2)), max log(p) variance = $(round(maximum([x.total_lp_var for x in result_mc.genq]), digits=2))"
    while min_neff_samps(result_mc.genq) < Neff_min
        old_xsamps = xsamps
        xsamps, logwts = update_samples(result_exact.x_obs, result_exact.sigma_obs, xsamps, logwts, result_mc.genq, Neff_min)
        for (i, (xs, oxs)) in enumerate(zip(xsamps, old_xsamps))
            if length(xs) != length(oxs)
                @info "Event $i: Added $(length(xs) - length(oxs)) samples"
            end
        end
        result_mc = run_sample_simulation(xsamps, logwts)
        @info "Found $(round(min_neff_samps(result_mc.genq), digits=2)) effective samples, max log(p) variance = $(round(maximum([x.total_lp_var for x in result_mc.genq]), digits=2))"
    end
    plot = pairplot_results(result_exact, result_mc)
    return (; result_exact = result_exact, result_mc = result_mc, plot = plot)
end

end # module HowManySamples
