"""
Main implementation of Normalized Cuts and Image Segmentation algorithm
Combined with insights from Multiclass Spectral Clustering
"""

export NormalizedCutsParams, compute_ncut_value, compute_image_affinity
export recursive_ncut, normalized_cuts_segmentation, compute_partition_cost

"""
    NormalizedCutsParams

Parameters structure for Normalized Cuts algorithm.

# Fields
- `σ_spatial::Float64`: Spatial scaling parameter for distance calculation
- `σ_feature::Float64`: Feature scaling parameter for similarity calculation
- `min_ncut::Float64`: Threshold for recursive partitioning (default: 0.1)
- `max_depth::Int`: Maximum recursion depth (default: 10)
- `k_nearest::Int`: Number of nearest neighbors for sparse affinity matrix
"""
struct NormalizedCutsParams
    σ_spatial::Float64
    σ_feature::Float64
    min_ncut::Float64
    max_depth::Int
    k_nearest::Int

    # Internal constructor to provide parameter validation
    function NormalizedCutsParams(σ_spatial::Float64, σ_feature::Float64, 
                                min_ncut::Float64, max_depth::Int, k_nearest::Int)
        σ_spatial > 0 || throw(ArgumentError("σ_spatial must be positive"))
        σ_feature > 0 || throw(ArgumentError("σ_feature must be positive"))
        min_ncut > 0 || throw(ArgumentError("min_ncut must be positive"))
        max_depth > 0 || throw(ArgumentError("max_depth must be positive"))
        k_nearest > 0 || throw(ArgumentError("k_nearest must be positive"))
        
        new(σ_spatial, σ_feature, min_ncut, max_depth, k_nearest)
    end
end

# External constructor, providing default values
function NormalizedCutsParams(;
    σ_spatial::Float64,
    σ_feature::Float64,
    min_ncut::Float64=0.1,
    max_depth::Int=10,
    k_nearest::Int=10)
    
    NormalizedCutsParams(σ_spatial, σ_feature, min_ncut, max_depth, k_nearest)
end


"""
    compute_image_affinity(spatial_coords::Matrix{Float64}, 
                         features::Matrix{Float64}, 
                         params::NormalizedCutsParams)

Constructs the affinity matrix for image segmentation considering both spatial and feature information.

# Arguments
- `spatial_coords::Matrix{Float64}`: N×2 matrix of pixel coordinates
- `features::Matrix{Float64}`: N×D matrix of pixel features (intensity, color, etc.)
- `params::NormalizedCutsParams`: Parameters for the algorithm

# Returns
- `W::Matrix{Float64}`: N×N sparse affinity matrix
"""
function compute_image_affinity(spatial_coords::Matrix{Float64}, 
                              features::Matrix{Float64}, 
                              params::NormalizedCutsParams)
    n = size(spatial_coords, 1)
    tree = KDTree(spatial_coords')
    W = zeros(Float64, n, n)
    
    for i in 1:n
        # Find k nearest neighbors in spatial domain
        idxs, dists = knn(tree, spatial_coords[i,:], params.k_nearest + 1)
        
        for j_idx in 2:length(idxs)
            j = idxs[j_idx]
            if i < j  # Compute only upper triangle
                # Spatial distance weight
                spatial_dist = dists[j_idx]
                spatial_weight = exp(-spatial_dist^2 / (2 * params.σ_spatial^2))
                
                # Feature distance weight
                feature_dist = norm(features[i,:] - features[j,:])
                feature_weight = exp(-feature_dist^2 / (2 * params.σ_feature^2))
                
                # Combined weight
                W[i,j] = W[j,i] = spatial_weight * feature_weight
            end
        end
    end
    
    return W
end


"""
    compute_ncut_value(W::Matrix{Float64}, partition::Vector{Int})

Computes the normalized cut value for a given partition.

# Arguments
- `W::Matrix{Float64}`: Affinity matrix
- `partition::Vector{Int}`: Binary partition vector (1s and 2s)

# Returns
- `ncut::Float64`: Normalized cut value
"""
function compute_ncut_value(W::Matrix{Float64}, partition::Vector{Int})
    A = findall(==(1), partition)
    B = findall(==(2), partition)
    
    # Compute cut value
    cut = sum(W[i,j] for i in A for j in B)
    
    # Compute associations
    assoc_A = sum(sum(W[i,:]) for i in A)
    assoc_B = sum(sum(W[i,:]) for i in B)
    
    return cut/assoc_A + cut/assoc_B
end


"""
    compute_partition_cost(V::Matrix{Float64}, partition::Vector{Int})

Computes the partition cost using the second smallest eigenvector.

# Arguments
- `V::Matrix{Float64}`: Matrix containing eigenvectors
- `partition::Vector{Int}`: Current partition assignment
- `R::Matrix{Float64}`: Optional rotation matrix for multiclass case

# Returns
- `cost::Float64`: Partition cost
"""
function compute_partition_cost(V::Matrix{Float64}, 
                              partition::Vector{Int},
                              R::Matrix{Float64}=Matrix{Float64}(I, size(V,2), size(V,2)))
    # Transform points if rotation matrix provided
    points = V * R
    
    # Compute centroids
    c1 = mean(points[partition .== 1, :], dims=1)
    c2 = mean(points[partition .== 2, :], dims=1)
    
    # Compute within-cluster distances
    cost = sum(norm(points[i,:] - (partition[i] == 1 ? c1 : c2))^2 
              for i in 1:size(points,1))
    
    return cost
end


"""
    recursive_ncut(W::Matrix{Float64}, params::NormalizedCutsParams, depth::Int=0)

Recursively applies normalized cuts to partition the graph.

# Arguments
- `W::Matrix{Float64}`: Affinity matrix
- `params::NormalizedCutsParams`: Algorithm parameters
- `depth::Int`: Current recursion depth

# Returns
- `partition::Vector{Int}`: Cluster assignments
"""
function recursive_ncut(W::Matrix{Float64}, 
                       params::NormalizedCutsParams, 
                       depth::Int=0)
    n = size(W, 1)
    
    # Base cases
    if depth >= params.max_depth || n <= 2
        return ones(Int, n)
    end
    
    # Compute normalized Laplacian
    L = compute_normalized_laplacian(W)
    
    # Get eigenvectors
    eigen_decomp = eigen(Symmetric(L))
    V = eigen_decomp.vectors[:, 1:2]  # Use first two eigenvectors
    
    # Try different splitting points
    best_partition = ones(Int, n)
    best_ncut = Inf
    
    # Use multiple initialization points for better results
    for init_method in [:kmeans, :threshold]
        if init_method == :kmeans
            result = kmeans(V', 2, init=:kmpp)
            partition = result.assignments
        else
            # Try different thresholds on second eigenvector
            v2 = V[:,2]
            thresholds = quantile(v2, [0.3, 0.4, 0.5, 0.6, 0.7])
            for t in thresholds
                partition = (v2 .>= t) .+ 1
                ncut = compute_ncut_value(W, partition)
                if ncut < best_ncut
                    best_ncut = ncut
                    best_partition = partition
                end
            end
        end
    end
    
    # If cut is good enough, recurse on subgraphs
    if best_ncut < params.min_ncut
        final_partition = copy(best_partition)
        
        # Recurse on each subgraph
        for label in [1,2]
            subset = findall(==(label), best_partition)
            if length(subset) > 1
                W_sub = W[subset, subset]
                sub_partition = recursive_ncut(W_sub, params, depth + 1)
                final_partition[subset] = sub_partition .+ maximum(final_partition)
            end
        end
        
        return final_partition
    end
    
    return best_partition
end


"""
    normalized_cuts_segmentation(spatial_coords::Matrix{Float64}, 
                               features::Matrix{Float64}, 
                               params::NormalizedCutsParams)

Main function for normalized cuts image segmentation.

# Arguments
- `spatial_coords::Matrix{Float64}`: N×2 matrix of pixel coordinates
- `features::Matrix{Float64}`: N×D matrix of pixel features
- `params::NormalizedCutsParams`: Algorithm parameters

# Returns
- `segments::Vector{Int}`: Segment labels for each pixel
- `W::Matrix{Float64}`: Computed affinity matrix
"""
function normalized_cuts_segmentation(spatial_coords::Matrix{Float64}, 
                                    features::Matrix{Float64}, 
                                    params::NormalizedCutsParams)
    # Compute affinity matrix
    W = compute_image_affinity(spatial_coords, features, params)
    
    # Apply recursive normalized cuts
    segments = recursive_ncut(W, params)
    
    return segments, W
end