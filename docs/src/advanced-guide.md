```@meta
CurrentModule = SpectralClusteringTools
```

```@index
```

# Spectral Clustering: Advanced Guide

## 1. Introduction

### 1.1 What is Spectral Clustering?

Spectral Clustering is a clustering method that utilizes eigenvalue decomposition of graphs to identify clusters in data. Unlike traditional clustering algorithms such as k-means, Spectral Clustering can capture nonlinear structures in the data.

The paper *"A Tutorial on Spectral Clustering"* provides a comprehensive theoretical foundation for Spectral Clustering, outlining the process as follows:

1. Construct a similarity graph
2. Compute the graph Laplacian
3. Perform eigenvalue decomposition
4. Embed data into a lower-dimensional space
5. Apply clustering (typically using k-means)

This tutorial will provide a guide on how to implement Spectral Clustering.

**Reference Papers:**

- [A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189)
- [Self-Tuning Spectral Clustering](https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf)

## 2. Parameter Settings (Parameter Types)

### 2.1 SpectralClusteringParams

**Overview:**
Defines the parameters for similarity graphs used in Spectral Clustering.

```julia
struct SpectralClusteringParams
    graph_type::Symbol  # :epsilon, :knn, :fully_connected
    k::Int  # Number of neighbors for k-NN graph
    ε::Float64  # Threshold for ε-neighborhood graph
    σ::Float64  # Scale for Gaussian similarity
end
```

### 2.2 SelfTuningParams

**Overview:**
Defines parameters for Self-Tuning Spectral Clustering.

```julia
struct SelfTuningParams
    k::Int  # Number of nearest neighbors
    σ::Float64  # Scale for Gaussian kernel
end
```

### 2.3 NormalizedCutsParams

**Overview:**
Defines parameters for the Normalized Cuts algorithm.

```julia
struct NormalizedCutsParams
    σ_spatial::Float64  # Spatial scale
    σ_feature::Float64  # Feature scale
    min_ncut::Float64  # Threshold for recursive splitting
    max_depth::Int  # Maximum recursion depth
    k_nearest::Int  # Number of nearest neighbors
end
```

## 3. Constructing a Similarity Graph

### 3.1 Types of Similarity Graphs

| Graph Type        | Description                              | Function                          |
|------------------|----------------------------------|----------------------------------|
| ε-neighborhood Graph | Connects points within distance ε | `epsilon_neighborhood_graph` |
| k-NN Graph       | Connects each point to k neighbors | `knn_graph`                     |
| Mutual k-NN Graph | Connects only mutual k-neighbors  | `mutual_knn_graph`              |
| Fully Connected Graph | Connects all points (weighted) | `fully_connected_graph`         |

## 4. Implementation of Spectral Clustering

### 4.1 Computing the Graph Laplacian

```julia
Lsym = compute_normalized_laplacian(W)
```

### 4.2 Eigenvalue Decomposition

```julia
eigen_decomp = eigen(Symmetric(Lsym))
U = eigen_decomp.vectors[:, 1:k]
```

### 4.3 Clustering using k-means

```julia
using Clustering
result = kmeans(U', 3, init=:kmpp)
clusters = result.assignments
```

## 5. Self-Tuning Spectral Clustering

```julia
W_self_tuning = construct_self_tuning_affinity(X, SelfTuningParams(3, 1.0))
clusters_self = self_tuning_spectral_clustering(W_self_tuning, 3)
```

## 6. Image Segmentation using Normalized Cuts

```julia
params = NormalizedCutsParams(σ_spatial=10.0, σ_feature=0.1)
W_image = compute_image_affinity(spatial_coords, features, params)
ncut_value = compute_ncut_value(W_image, partition)
segments = recursive_ncut(W_image, params)
segments, W_final = normalized_cuts_segmentation(spatial_coords, features, params)
```

## 7. Conclusion

This tutorial has provided an overview of the theory behind Spectral Clustering and its implementation in Julia.

| Task                 | Function                              |
|----------------------|--------------------------------------|
| Construct Similarity Graph | `construct_similarity_graph`       |
| k-NN Graph          | `knn_graph`                           |
| Mutual k-NN Graph   | `mutual_knn_graph`                    |
| Fully Connected Graph | `fully_connected_graph`             |
| Compute Normalized Laplacian | `compute_normalized_laplacian` |
| Spectral Clustering | `spectral_clustering`                 |
| Self-Tuning Clustering | `self_tuning_spectral_clustering`   |
| Image Segmentation  | `normalized_cuts_segmentation`        |

Use this guide to explore advanced clustering techniques and image segmentation research.
