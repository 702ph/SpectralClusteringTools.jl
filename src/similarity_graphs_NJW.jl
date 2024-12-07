"""
1. Construct a similarity graph. W: weighted adjacency matrix(A Tutorial on Spectral Clustering P7)
● According to the reference, there are three ways to do it:
    (1) The ε-neighborhood graph
    (2) k-nearest neighbor graphs
    (3) The fully connected graph
● Parameters for Spectral clustering
"""
struct  SpectralClusteringParams
    graph_type::Symbol # choose which method to build the similarity graph
    k::Int      # k-nearest neighbors
    ε::Float64  # ε (Epsilon) neighborhood
    σ::Float64  # σ (sigma)-->Gaussian similarity
end

"""
● (Build) The ε-neighborhood graph 
●  Connect all points whose pairwise distances are smaller than ε
●  Considered as an unweighted graph (weight is either 0 or 1)
●  But since W is weighted in NJW, ε-neighborhood may not be used for it, just for short try, and compare the result
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
● (Construct) k-nearest neighbor graphs
● Each vertex is connected to its k nearest neighbors (Connect vertex vi with vertex vj if vj is among the k-nearest neighbors of vi)
● Two ways: the k-nearest neighbor graph && mutual k-nearest neighbor graph
"""

"""
● k-nearest neighbor graph
● Ignore the directions of the edges
● Connect vi and vj with an undirected edge if vi is among the k-nearest neighbors of vj, vice versa
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
● (Construct) Mutual k-nearest neighbor graph
● Use Local scaling as mentioned in Self Tuning?
● Vertices are connected only if they are among each other's k nearest neighbors
● Connect vertices vi and vj if both vi is among the k-nearest neighbors of vj and vj is among the k-nearest neighbors of vi
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
● Construct fully connected graph
● Connect all points with positive similarity with each other
● Weight all edge by sij
● Only useful if the similarity function itself models local neighbors， example: Gaussian
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