"""
SpectralClusteringTools.jl
Main module file for spectral clustering implementations
"""
module SpectralClusteringTools

# 1. Import packages
using LinearAlgebra: norm, eigvals, eigvecs, Diagonal, diag, Symmetric, eigen, I, dot 
using Statistics: mean, std, median
using NearestNeighbors: KDTree, knn
using Clustering: kmeans
using ArnoldiMethod
using SparseArrays

# 2. Include implementations
# Basic utilities
include("Utils/DataSets.jl")

# Self-tuning spectral clustering
include("self_tuning.jl")

# Traditional spectral clustering
include("similarity_graphs_NJW.jl")
include("affinity_matrix_NJW.jl")
include("spectral_clustering_NJW.jl")

timestwo(x) = 2 * x

# 3. Exports
export 
    # Data generation
    make_sphere,
     
    # Parameter types
    SpectralClusteringParams,
    SelfTuningParams,  
    
    # NJW spectral clustering
    construct_similarity_graph,
    spectral_clustering,
    epsilon_neighborhood_graph,
    knn_graph, 
    mutual_knn_graph,
    fully_connected_graph,
    
    # Self-tuning spectral clustering
    construct_self_tuning_affinity,
    self_tuning_spectral_clustering,
    timestwo
    

end # module