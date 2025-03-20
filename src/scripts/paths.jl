# Absolute path to the top level of the repository
root = joinpath(@__DIR__, "..", "..")

# Absolute path to the `src` folder
src = joinpath(root, "src")

# Absolute path to the `src/data` folder (contains datasets)
data = joinpath(src, "data")

# Absolute path to the `src/static` folder (contains static images)
static = joinpath(src, "static")

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = joinpath(src, "scripts")

# Absolute path to the HowManySamples pkg
hmspkg = joinpath(scripts, "HowManySamples")

# Absolute path to the `src/tex` folder (contains the manuscript)
tex = joinpath(src, "tex")

# Absolute path to the `src/tex/figures` folder (contains figure output)
figures = joinpath(tex, "figures")

# Absolute path to the `src/tex/output` folder (contains other user-defined output)
output = joinpath(tex, "output")
