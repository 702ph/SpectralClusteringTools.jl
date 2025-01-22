module SpectralClusteringTools


using LinearAlgebra: norm, eigvals, eigvecs, Diagonal, diag, Symmetric, eigen
using Statistics: mean, median
using NearestNeighbors: KDTree, knn
using Clustering: kmeans


# import source files
include("Utils/DataSets.jl")
include("similarity_graphs_NJW.jl")
include("affinity_matrix_NJW.jl")
include("spectral_clustering_NJW.jl")


#export public functions
export SpectralClusteringParams, construct_similarity_graph,
       spectral_clustering, epsilon_neighborhood_graph,
       knn_graph, mutual_knn_graph,
       fully_connected_graph
       
end
