import os

os.environ["JULIA_PROJECT"] = "."

envvars:
    "JULIA_PROJECT"

rule Manifest:
    input:
        "Project.toml"
    output:
        "Manifest.toml"
    shell:
        "julia -e 'using Pkg; Pkg.instantiate()'"

rule traces:
    input:
        "Manifest.toml",
        "src/scripts/generate_traces.jl"
    output:
        "src/data/traces.h5"
    shell:
        "julia src/scripts/generate_traces.jl"

rule min_neff:
    input:
        "Manifest.toml",
        "src/scripts/generate_min_neff.jl",
        "src/data/traces.h5"
    output:
        "src/tex/output/min_neff.txt",
        "src/tex/output/max_lp_var.txt"
    shell:
        "julia src/scripts/generate_min_neff.jl"