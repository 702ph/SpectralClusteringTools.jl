"""
SpectralClusteringTools.jl
Main module file for spectral clustering implementations
"""
module SpectralClusteringTools

# 1. Import packages
using LinearAlgebra: norm, eigvals, eigvecs, Diagonal, diag, Symmetric, eigen, I, dot 
using Statistics: mean, std, median, quantile
using NearestNeighbors: KDTree, knn
using Clustering: kmeans
using Plots

# 2. Include implementations

# Self-tuning spectral clustering
include("self_tuning.jl")

# Traditional spectral clustering
include("similarity_graphs_NJW.jl")
include("affinity_matrix_NJW.jl")
include("spectral_clustering_NJW.jl")

# Normalized cuts and segmentation
include("normalized_cuts.jl")

# Basic utilities
include("Utils/DataSets.jl")
include("Utils/DataSets_2D.jl")
include("Utils/compare_results.jl")
include("Utils/2D_data_testing.jl")


# 3. Exports
export 
     
    # Parameter types
    SpectralClusteringParams,
    SelfTuningParams,  
    NormalizedCutsParams, 
    
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

    # Normalized cuts and segmentation
    normalized_cuts_segmentation,
    recursive_ncut,
    compute_ncut_value,
    compute_image_affinity,

    # Data generation
    make_sphere,
    generate_mixed_concentric_data,
    generate_mixed_moons_data,
    make_circles,
    make_lines_2d,
    make_spirals_2d,
    make_blobs_2d

end # module