export self_tuning_spectral_clustering
"""
Parameters for Self-Tuning Spectral Clustering
"""
struct SelfTuningParams
    K::Int      # K-nearest neighbors for local scaling (default=7 as mentioned in the paper)
    use_local_scaling::Bool  # whether to use local scaling
end


struct GivensRotation
    i::Int       # row index
    j::Int       # column index
    θ::Float64   # rotation angle
end

"""
Compute local scaling parameters σᵢ for each point
σᵢ = d(sᵢ, s_K) where s_K is the K-th neighbor of point sᵢ
"""
function compute_local_scaling(X::Matrix{Float64}, K::Int)
    n = size(X, 2)
    tree = KDTree(X)
    #σ = zeros(Float64, n)

    neighbors_and_dists = [knn(tree, X[:,i], K + 1) for i in 1:n]
    distances = [dists[2:end] for (_, dists) in neighbors_and_dists]
    σ = [dists[end] for dists in distances]
    
    eps_stable = 1e-10
    median_σ = median(σ)
    σ = [max(s, eps_stable * median_σ) for s in σ]

    return σ, Dict(
        "tree" => tree,
        "distances" => distances,
        "neighbors_and_dists" => neighbors_and_dists
    )
end

"""
Construct self-tuning affinity matrix that can handle both spherical data and concentric circles.
Aᵢⱼ = exp(-d²(sᵢ,sⱼ)/(σᵢσⱼ))
Key features:
1. Adaptive local scaling parameter computation
2. Support for both spherical and Euclidean distances
3. Enhanced connectivity for circular structures
"""
function construct_self_tuning_affinity(X::Matrix{Float64}, params::SelfTuningParams; 
                                      is_spherical::Bool=false)
    n = size(X, 2)
    A = zeros(Float64, n, n)
    
    # 1. Normalize points if using spherical distance
    if is_spherical
        X_normalized = X ./ sqrt.(sum(X.^2, dims=1))
    end
    
    # 2. Compute pairwise distances
    distances = zeros(Float64, n, n)
    for i in 1:n
        for j in (i+1):n
            if is_spherical
                # Use great circle distance for spherical data
                X_i = X_normalized[:,i]
                X_j = X_normalized[:,j]
                dist = 2.0 * asin(min(1.0, norm(X_i - X_j) / 2.0))
            else
                # Use Euclidean distance for planar data
                dist = norm(X[:,i] - X[:,j])
            end
            distances[i,j] = distances[j,i] = dist
        end
    end
    
    # 3. Compute adaptive local scaling parameters
    σ = zeros(Float64, n)
    K_adaptive = zeros(Int, n)
    
    for i in 1:n
        # Sort distances for this point
        sorted_dists = sort(distances[i,:])
        
        # Find local density estimate
        local_density = mean(sorted_dists[2:params.K])
        
        # Adjust K based on local density and data type
        if is_spherical
            # For spherical data, use fixed K
            K_effective = params.K
        else
            # For planar data (e.g., concentric circles), allow larger neighborhoods
            K_effective = min(2 * params.K, n-1)
        end
        
        # Use the distance to the K-th neighbor as sigma
        σ[i] = sorted_dists[K_effective + 1]
        K_adaptive[i] = K_effective
    end
    
    # 4. Compute affinities with appropriate connectivity
    for i in 1:n
        for j in (i+1):n
            dist = distances[i,j]
            
            # Compute local scaling factor
            local_scale = sqrt(σ[i] * σ[j])
            
            # Enhanced affinity computation
            if is_spherical
                # For spherical data, use standard scaling
                scaled_dist = (dist^2) / (local_scale^2)
            else
                # For concentric circles, use enhanced scaling
                global_scale = median(σ)
                scaled_dist = (dist^2) / (local_scale * global_scale)
            end
            
            affinity = exp(-scaled_dist)
            
            # Apply appropriate threshold
            threshold = is_spherical ? 0.001 : 0.01
            if affinity > threshold
                A[i,j] = A[j,i] = affinity
            end
        end
    end
    
    # 5. Enhance local connectivity if dealing with concentric circles
    if !is_spherical
        A_enhanced = copy(A)
        for i in 1:n
            # Find top K neighbors
            neighbors = partialsortperm(distances[i,:], 1:K_adaptive[i])
            for j in neighbors
                if A[i,j] > 0
                    # Strengthen connections within local neighborhood
                    A_enhanced[i,j] = A_enhanced[j,i] = max(A[i,j], 0.1)
                end
            end
        end
        return A_enhanced
    end
    
    return A
end

"""
Apply Givens rotation to a matrix
"""
function apply_givens_rotation!(X::Matrix{Float64}, G::GivensRotation)
    c = cos(G.θ)
    s = sin(G.θ)
    
    for k in 1:size(X, 2)
        xi = X[G.i, k]
        xj = X[G.j, k]
        X[G.i, k] = c * xi - s * xj
        X[G.j, k] = s * xi + c * xj
    end
end


"""
Recover the rotation matrix R which best aligns X's columns with the canonical coordinate
system using incremental gradient descent scheme.

Based on Appendix A of the paper:
1. Uses Givens rotations for minimal parameterization
2. Implements gradient descent to minimize the cost function
3. Updates rotation incrementally

Parameters:
- X: Matrix of eigenvectors to be aligned
- max_iter: Maximum number of iterations for gradient descent
- tol: Tolerance for convergence
- α: Learning rate for gradient descent

Returns:
- R: The optimal rotation matrix
- Z: The aligned matrix (Z = XR)
- cost: Final cost value
"""

function recover_rotation(X::Matrix{Float64}; 
                    max_iter::Int=2000, 
                    tol::Float64=1e-6,
                    α::Float64=0.1)
    n, C = size(X)
    if C == 1
        return Matrix{Float64}(I, 1, 1), X, 0.0
    end

    # Initialize rotation matrix
    R = Matrix{Float64}(I, C, C)
    Z = copy(X)

    # Create index pairs for Givens rotations
    index_pairs = [(i,j) for i in 1:C-1 for j in i+1:C]
    K = length(index_pairs)

    # Initialize rotation angles
    Θ = zeros(Float64, K)

    # Gradient descent
    prev_cost = Inf
    converged = false

    for iter in 1:max_iter
        # 1. Compute current cost
        Mi = zeros(Float64, n)
        for i in 1:n
            Mi[i] = maximum(abs.(Z[i,:]))
            if Mi[i] < eps(Float64)
                Mi[i] = 1.0  # Avoid division by zero
            end
        end

        current_cost = sum(sum(Z[i,j]^2/Mi[i]^2 for j in 1:C) for i in 1:n)

        # Check convergence
        if abs(current_cost - prev_cost) < tol
            converged = true
            break
        end
        prev_cost = current_cost

        # 2. Update each Givens rotation
        for (k, (i,j)) in enumerate(index_pairs)
            gradient = 0.0
            for row in 1:n
                if Mi[row] > eps(Float64)
                    gradient += (Z[row,i]^2 - Z[row,j]^2) / Mi[row]^2
                end
            end

            # Update angle with gradient descent
            Θ[k] -= α * gradient

            # Apply rotation
            c, s = cos(Θ[k]), sin(Θ[k])
            for col in 1:C
                zi = Z[:,col][i]
                zj = Z[:,col][j]
                Z[:,col][i] = c * zi - s * zj
                Z[:,col][j] = s * zi + c * zj
            end

            # Update rotation matrix
            R_temp = Matrix{Float64}(I, C, C)
            R_temp[i,i] = R_temp[j,j] = c
            R_temp[i,j] = -s
            R_temp[j,i] = s
            R = R * R_temp
        end
    end

    if !converged
    @warn "Rotation recovery did not converge"
    end

    return R, Z, prev_cost
end


"""
Determine optimal number of clusters based on eigenvalues and eigenvectors alignment
using average cost comparison method.
"""
function determine_best_C(eigvals::Vector{Float64}, eigvecs::Matrix{Float64}, max_C::Int)
    normalized_eigvals = eigvals ./ maximum(eigvals)
    sorted_indices = sortperm(normalized_eigvals, rev=true)
    sorted_eigvecs = eigvecs[:, sorted_indices]
    
    costs = Float64[]
    rotations = []
    
    # Calculate costs for each potential number of clusters
    for C in 2:max_C
        X = sorted_eigvecs[:, 1:C]
        R, Z, cost = recover_rotation(X)
        push!(costs, cost)
        push!(rotations, (R, Z))
    end
    
    # Calculate average cost
    avg_cost = mean(costs)
    
    # Find the number of clusters whose cost is closest to the average
    cost_differences = abs.(costs .- avg_cost)
    best_index = argmin(cost_differences)
    best_C = best_index + 1  # Add 1 because we started from C=2
    
    # Get corresponding rotation for best C
    best_R, best_Z = rotations[best_index]
    
    # Print analysis for debugging
    println("\nCost Analysis for Cluster Selection:")
    println("Average cost across all C: ", round(avg_cost, digits=4))
    println("Cost differences from average:")
    for (i, diff) in enumerate(cost_differences)
        println("C = $(i+1): diff = $(round(diff, digits=4))")
    end
    println("Selected C = $best_C with smallest difference from average")
    
    return best_C, best_R, best_Z
end


"""
Analyze eigengaps to determine the optimal number of clusters and corresponding eigenvectors.
Based on the theory that:
1. For ideal case, eigenvalue 1 should be repeated C times (C = number of clusters)
2. There should be a significant gap between the C-th and (C+1)-th eigenvalues
3. The first C eigenvectors should correspond to cluster indicators

Returns:
- X: Matrix of selected eigenvectors (before rotation)
- best_C: Optimal number of clusters
- analysis_info: Dictionary containing analysis details
"""
function analyze_eigengaps(L::Matrix{Float64}, max_C::Int)
    
    eigen_decomp = eigen(Symmetric(L))
    eigvals = eigen_decomp.values
    eigvecs = eigen_decomp.vectors
    
    best_C, R, Z = determine_best_C(eigvals, eigvecs, max_C)
    
    costs = Float64[]
    for C in 2:max_C
        X_C = eigvecs[:, 1:C]
        _, _, cost = recover_rotation(X_C)
        push!(costs, cost)
    end
    
    normalized_eigvals = eigvals ./ maximum(eigvals)
    sorted_indices = sortperm(normalized_eigvals, rev=true)
    sorted_eigvals = normalized_eigvals[sorted_indices]
    
    analysis_info = Dict{String, Any}(
        "normalized_eigenvalues" => sorted_eigvals,
        "rotation_matrix" => R,
        "aligned_vectors" => Z,
        "costs_per_C" => costs,
        "rotation_cost" => costs[best_C-1],  
        "final_alignment_quality" => maximum(abs.(Z), dims=2)  # 每行的最大值
    )
    
    println("\nDetailed Analysis Results:")
    println("------------------------")
    println("Eigenvalue Analysis:")
    println("- Top 5 eigenvalues: ", round.(sorted_eigvals[1:min(5, end)], digits=4))
    println("- Relative magnitudes: ", round.(sorted_eigvals[1:min(5, end)] ./ sorted_eigvals[1], digits=4))
    
    println("\nClustering Analysis:")
    println("- Best number of clusters: ", best_C)
    println("- Cost values for different C:")
    for (c, cost) in enumerate(costs)
        println("  C = $(c+1): cost = $(round(cost, digits=4))")
    end
    
    println("\nRotation Analysis:")
    println("- Final rotation cost: ", round(costs[best_C-1], digits=4))
    println("- Average alignment quality: ", 
            round(mean(maximum(abs.(Z), dims=2)), digits=4))
    
    println("\nAlignment Properties:")
    println("- Max absolute values per cluster:")
    for i in 1:size(Z, 2)
        println("  Cluster $i: ", round(maximum(abs.(Z[:,i])), digits=4))
    end
    
    println("------------------------")
    
    return Z, best_C, analysis_info
end

"""
Self-tuning spectral clustering main function

This implementation follows the algorithm described in the paper:
1. Compute local scaling parameters
2. Construct affinity matrix using local scaling
3. Compute normalized Laplacian
4. Find eigenvectors and determine optimal number of clusters
5. Recover rotation matrix for alignment
6. Perform final clustering

Parameters:
- X: Input data matrix
- max_C: Maximum number of clusters to consider
- params: Self-tuning parameters (K for local scaling)

Returns:
- assignments: Cluster assignments for each data point
- best_C: Optimal number of clusters
- analysis_info: Detailed analysis information
"""

function self_tuning_spectral_clustering(X::Matrix{Float64}, 
                                       max_C::Int,
                                       params::SelfTuningParams;
                                       is_spherical::Bool=false
                                      )
    # Parameter validation
    max_C <= 0 && throw(ArgumentError("Maximum number of clusters must be positive"))
    params.K <= 0 && throw(ArgumentError("K for local scaling must be positive"))

    try
        println("Step 1: Constructing affinity matrix with local scaling...")
        # 1. Construct affinity matrix with local scaling
        A = construct_self_tuning_affinity(X, params, is_spherical=is_spherical)
        
        println("Step 2-3: Computing normalized Laplacian matrix...")
        # 2. Compute the degree matrix D
        eps_stable = 1e-10
        D_values = sum(A, dims=2)[:]
        D_values = [d + eps_stable for d in D_values]  # Add small epsilon to each degree
        D = Diagonal(D_values)
        
        # 3. Compute normalized affinity matrix L = D^(-1/2)AD^(-1/2)
        D_inv_sqrt = Diagonal(1.0 ./ sqrt.(diag(D)))

        # Check for any invalid values
        if any(isnan, D_inv_sqrt.diag) || any(isinf, D_inv_sqrt.diag)
            throw(ArgumentError("Invalid values in degree matrix"))
        end

        L = D_inv_sqrt * A * D_inv_sqrt

        # # Ensure symmetry
        L = (L + L') / 2
        
        # # Ensure numerical stability
        L = clamp.(L, -1.0 + eps_stable, 1.0 - eps_stable)
        
        println("Step 4: Analyzing eigengaps and determining optimal clusters...")
        # 4. Find eigenvectors and determine optimal clusters
        eigvec_matrix, best_C, analysis_info = analyze_eigengaps(
            L, 
            max_C
        )
        
        println("Step 5: Recovering optimal rotation...")
        # 5. Recover optimal rotation
        R, Z, rotation_cost = recover_rotation(eigvec_matrix)
        
        analysis_info["rotation_matrix"] = R
        analysis_info["rotation_cost"] = rotation_cost
        analysis_info["original_eigenvectors"] = eigvec_matrix
        println("Rotation cost: ", round(rotation_cost, digits=4))
        
        println("Step 6: Normalizing rotated eigenvectors...")
        norms = sqrt.(sum(Z.^2, dims=2))
        norms = [n + eps_stable for n in norms] 
        T = Z ./ norms
        
        println("Step 7: Performing k-means clustering...")
        # k-means clustering initialization
        best_result = nothing
        min_cost = Inf
        
        # Perform k-means
        for i in 1:10
            result = kmeans(T', best_C;
                          maxiter=500,
                          display=:none,
                          init=:kmpp,
                          tol=1e-6)
            
            if result.totalcost < min_cost
                min_cost = result.totalcost
                best_result = result
            end
        end
        
        analysis_info["clustering_cost"] = min_cost
        analysis_info["normalized_vectors"] = T
        
        println("Clustering completed successfully!")
        println("Number of clusters found: ", best_C)
        println("Final clustering cost: ", round(min_cost, digits=4))
        
        return best_result.assignments, best_C, analysis_info
        
    catch e
        @warn "Error in self-tuning spectral clustering" exception=(e, catch_backtrace())
        rethrow(e)
    end
end