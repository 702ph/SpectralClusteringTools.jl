
"""
    struct SpectralClusteringParams

        A structure for defining parameters for Spectral Clustering.
# Fields
- `graph_type::Symbol` : The method used to construct the similarity graph (`:epsilon`, `:knn`, `:fully_connected`)
- `k::Int` : Number of nearest neighbors (used when `:knn` is selected)
- `ε::Float64` : Epsilon neighborhood (used when `:epsilon` is selected)
- `σ::Float64` : Parameter for the Gaussian similarity function
# Methods for constructing the similarity graph
1. `:epsilon` - ε-neighborhood graph
2. `:knn` - k-nearest neighbor graph
3. `:fully_connected` - Fully connected graph
Reference: "A Tutorial on Spectral Clustering" (P7)
"""
struct  SpectralClusteringParams
    graph_type::Symbol # choose which method to build the similarity graph
    k::Int      # k-nearest neighbors
    ε::Float64  # ε (Epsilon) neighborhood
    σ::Float64  # σ (sigma)-->Gaussian similarity
end


"""
    epsilon_neighborhood_graph(X::Matrix{Float64}, ε::Float64)

Constructs an ε-neighborhood graph, where an edge is formed between two points if their Euclidean distance is less than ε.

# Arguments
- `X`: A matrix where each column represents a data point in a feature space.
- `ε`: A threshold distance; two points are connected if their Euclidean distance is smaller than `ε`.

# Returns
- An adjacency matrix `W` of size `(n, n)`, where `n` is the number of data points.
  - `W[i, j] = 1.0` if `||X[:, i] - X[:, j]||_2 < ε`, otherwise `W[i, j] = 0.0`.
  - The matrix `W` is symmetric.

# Notes
- The graph is unweighted (binary adjacency matrix).
- This method is mainly used for comparison, as NJW spectral clustering typically uses a weighted similarity graph.

# Example
```julia
X = rand(2, 5)  # 2D points (each column is a data point)
ε = 0.5
W = epsilon_neighborhood_graph(X, ε)
```
"""
function epsilon_neighborhood_graph(X::Matrix{Float64}, ε::Float64)
    # size(X,1)--> rows(feature) ; size(X,2)-->columns(datapoints)
    n = size(X, 2)
    W = zeros(Float64, n, n) # adjacency matrix：store whether two points are connected or not

    for i in 1:n
        for j in (i+1):n
            # calculate L2-norm (Euclidean distance) between point i and point j
            distance = norm(X[:, i] - X[:, j])  # norm(A, p::Real=2), default p=2
            if distance < ε
                W[i,j] = W[j,i] =1.0
            end
        end
    end

    return W
end


"""
    knn_graph(X::Matrix{Float64}, k::Int)

Constructs a k-nearest neighbor (k-NN) graph, where each data point is connected to its `k` nearest neighbors. The graph is **undirected**, meaning that if `vi` is among the `k` nearest neighbors of `vj`, then `vj` is also connected to `vi`.

# Arguments
- `X`: A matrix where each column represents a data point in a feature space.
- `k`: The number of nearest neighbors to consider for each data point.

# Returns
- An adjacency matrix `W` of size `(n, n)`, where `n` is the number of data points.
  - If two points are mutual k-nearest neighbors, an edge is formed.
  - `W[i, j]` represents the similarity between points `i` and `j`, computed using a Gaussian similarity function:

    ```math
    W[i, j] = \exp\\left(-\frac{d_{ij}^2}{\\sigma_i \\sigma_j}\right)
    ```

  - `d_{ij}` is the Euclidean distance between points `i` and `j`.
  - `σ_i` and `σ_j` are local scaling parameters based on the k-th nearest neighbor distance.
  - The matrix `W` is symmetric.

# Notes
- A **KDTree** is used for efficient nearest neighbor search.
- The graph is **undirected**, meaning that if `vi` is in the k-nearest neighbors of `vj`, then an edge is also created from `vj` to `vi`.
- **Local scaling** is applied using the k-th nearest neighbor distance to prevent numerical instability.
- The **median of the k-th nearest neighbor distances** is used to avoid division by very small values.

# Example
```julia
    using NearestNeighbors  # Ensure NearestNeighbors.jl is installed for KDTree
    X = rand(2, 10)  # 2D points (each column is a data point)
    k = 3
    W = knn_graph(X, k)
```
"""
function knn_graph(X::Matrix{Float64}, k::Int)
    n = size(X, 2)
    # KDTree: Space-partitioning data structure for organizing points in a k-dimensional space
    # Efficiently find nearest neighbors
    tree = KDTree(X) 
    W = zeros(Float64, n, n) # adjacency matrix 

    neighbors_and_dists = [knn(tree, X[:,i], k+1) for i in 1:n]
    distances = [dists[2:end] for (_, dists) in neighbors_and_dists]

    
    for i in 1:n
        # For every data point, using knn find its k+1 nearest neighbors --> k+1 cause it includes the point istelf
        # Returning: Index of nearest neighbors (idxs) && corresponding distance (dists)
        idxs, dists = knn(tree, X[:, i], k+1) 

        #for j in idxs[2:end] # Skip the first point (the point itself)
        #for idx_pos in idxs[2:end]

        eps_stable = 1e-10 #Avoid σi σj too small, W[i,j] divide zero
        median_dist = median([d[end] for d in distances])

        for idx_pos in 2:length(idxs)
            #W[i,j] = W[j,i]=1.0
            j = idxs[idx_pos]
            dist = dists[idx_pos]

            σi = max(distances[i][end], eps_stable * median_dist)  # Apply local scaling based on k-th nearest neighbor distance
            σj = max(distances[j][end], eps_stable * median_dist)

            similarity = exp(-dist^2 / (σi * σj))
            #similarity = exp(-dist^2 / 2)
            W[i,j] = W[j,i] = similarity
        end
    end
    return W
end


"""
    mutual_knn_graph(X::Matrix{Float64}, k::Int)

Constructs a **mutual k-nearest neighbor (k-NN) graph**, where two data points `vi` and `vj` are connected **only if they are among each other's k nearest neighbors**. The edge weights are computed using a **Gaussian similarity function with local scaling**.

# Arguments
- `X`: A matrix where each column represents a data point in a feature space.
- `k`: The number of nearest neighbors to consider for each data point.

# Returns
- An adjacency matrix `W` of size `(n, n)`, where `n` is the number of data points.
  - `W[i, j] > 0` only if `vi` is in the k-nearest neighbors of `vj` **and** `vj` is in the k-nearest neighbors of `vi`.
  - The edge weight is computed using a Gaussian similarity function with **adaptive local scaling**:

    ``W[i, j] = \exp\\left(-\frac{d_{ij}^2}{\\sigma_i \\sigma_j}\right)``

  - `d_{ij}` is the Euclidean distance between points `i` and `j`.
  - `σ_i` and `σ_j` are local scaling parameters based on the k-th nearest neighbor distance.
  - The matrix `W` is symmetric.

# Notes
- **Mutual k-NN condition**: Unlike standard k-NN graphs, an edge is only formed if `vi` and `vj` are in each other's k-nearest neighbor lists.
- **Local scaling**: The similarity measure adapts based on the k-th nearest neighbor distance of each point (`σ_i` and `σ_j`).
- **KDTree** is used for efficient nearest neighbor search.

# Example
```julia
    using NearestNeighbors  # Ensure NearestNeighbors.jl is installed for KDTree
    X = rand(2, 10)  # 2D points (each column is a data point)
    k = 3
    W = mutual_knn_graph(X, k)
````
"""
function mutual_knn_graph(X::Matrix{Float64}, k::Int)
    n = size(X, 2)
    tree = KDTree(X)
    W = zeros(Float64, n, n)
    
    # Find and store k-nearest neighbors for every data point
    neighbors_and_dists = [knn(tree, X[:,i], k+1) for i in 1:n]
    neighbors = [Set(idxs[2:end]) for (idxs, _) in neighbors_and_dists]
    distances = [dists[2:end] for (_, dists) in neighbors_and_dists]

    # Error existing when simply use 1 and 0
    # for i in 1:n
    #     for j in (i+1):n
    #         if j in neighbors[i] && i in neighbors
    #             W[i,j] = W[j,i] =1.0
    #         end
    #     end
    # end

    for i in 1:n
        for j in (i+1):n
            if j in neighbors[i] && i in neighbors[j]
                # Use a Gaussian kernel to compute weights rather than a simple 1/0 connection
                dist = norm(X[:,i] - X[:,j])
                # Use adaptive local scaling parameters
                σi = distances[i][end] # Apply local scaling based on k-th nearest neighbor distance
                σj = distances[j][end]

                W[i,j] = W[j,i] = exp(-dist^2 / (σi * σj))
            end
        end
    end

    return W
end


"""
    fully_connected_graph(X::Matrix{Float64}, σ::Float64)

Constructs a **fully connected graph**, where all data points are connected to each other with an edge. The edge weights are determined by a similarity function, such as a **Gaussian kernel**.

# Arguments
- `X`: A matrix where each column represents a data point in a feature space.
- `σ`: A scaling parameter for the Gaussian similarity function.

# Returns
- An adjacency matrix `W` of size `(n, n)`, where `n` is the number of data points.
  - `W[i, j]` represents the similarity between points `i` and `j`, computed as:

    ``W[i, j] = \exp\\left(-\frac{||X_i - X_j||^2}{2\\sigma^2}\right)``

  - `W[i, j]` is always positive and symmetric (`W[i, j] = W[j, i]`).
  - Larger values of `σ` result in more globally connected graphs, while smaller `σ` emphasizes local neighborhoods.

# Notes
- **All points are connected**: Unlike k-NN or ε-neighborhood graphs, this graph is fully connected.
- **Edge weights depend on similarity**: The similarity function should naturally model local neighborhoods (e.g., the Gaussian kernel).
- **Gaussian kernel ensures smooth transitions**: The similarity decreases exponentially as the distance increases.
- **No thresholding**: Any two points with positive similarity are connected.

# Example
```julia
    X = rand(2, 10)  # 2D points (each column is a data point)
    σ = 1.0
    W = fully_connected_graph(X, σ)
```
"""
function fully_connected_graph(X::Matrix{Float64}, σ::Float64)
    n = size(X, 2)
    W = zeros(Float64, n, n)
    
    for i in 1:n
        for j in (i+1):n
            # s(xi, xj) = exp(-||xi - xj||² / (2σ²))
            dist = norm(X[:,i] - X[:,j])
            similarity = exp(-dist^2 / (2 * σ^2))
            W[i,j] = W[j,i] = similarity
        end
    end
    
    return W
end