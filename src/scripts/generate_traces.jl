using HowManySamples
using HDF5
using MCMCChainsStorage
using Random

include("paths.jl")

Random.seed!(0x23bc851a9f236201)
result = run_one_simulation()

exact_trace = result.result_exact.trace
mc_trace = append_generated_quantities(result.result_mc.trace, result.result_mc.genq)

h5open(joinpath(data, "traces.h5"), "w") do file
    exact_group = create_group(file, "exact")
    mc_group = create_group(file, "mc")

    write(exact_group, exact_trace)
    write(mc_group, mc_trace)
end