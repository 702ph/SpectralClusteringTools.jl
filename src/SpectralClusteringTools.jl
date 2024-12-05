module SpectralClusteringTools


include("Utils/DataSets.jl")
include("Utils/Visualization.jl")

"""
    timestwo(x)

Multiply the input `x` by two.
"""
timestwo(x) = 2 * x

export timestwo

end
