import os

os.environ["JULIA_PROJECT"] = "."

envvars:
    "JULIA_PROJECT"

rule Manifest:
    input:
        "Project.toml"
    output:
        "Manifest.toml"
    script:
        "julia -e 'using Pkg; Pkg.instantiate()'"