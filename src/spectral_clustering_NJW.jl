"""
Main Function for NJW spectral Clustering
"""

function spectral_clustering(X::Matrix{Float64}, 
    k::Int, 
    params::SpectralClusteringParams)
    # Parameter validation
    k <= 0 && throw(ArgumentError("Number of clusters must be positive"))
    params.σ <= 0 && throw(ArgumentError("Sigma σ must be positive"))
    params.k <= 0 && throw(ArgumentError("k for knn must be positive"))

    
    try
        # 1. Construct similarity graph
        W = if params.graph_type == :ε
            epsilon_neighborhood_graph(X, params.ε)
        elseif params.graph_type == :knn
            knn_graph(X, params.k)
        elseif params.graph_type == :mutual_knn
            mutual_knn_graph(X, params.k)
        else
            fully_connected_graph(X, params.σ)
        end

        # Check if W is valid
        if !any(W .!= 0)
            throw(ArgumentError("Similarity matrix is all zeros"))
        end

        # 2. Compute normalized Laplacian
        L = compute_normalized_laplacian(W)

        # 3. Compute the first(top) k eigenvectors u1, . . . , uk of Lsym.
        eigen_decomp = eigen(Symmetric(L)) # compute eigenvalues, and eigenvectors
        eigvals_L, eigvecs_L = eigen_decomp.values, eigen_decomp.vectors
            # ● partialsortperm: Find the location of the largest k eigenvalues
            # ● rev=true：descending order
        _, top_k_indices = partialsortperm(eigvals_L, 1:k, rev=true) 
        U = eigvecs_L[:, top_k_indices] # From all the eigenvectors the k corresponding to the largest eigenvalue is selected
        

        # 4. Normalize rows to unit length
        # ● Form the matrix T ∈ R^(n×k) from U by normalizing the rows to norm 1
        # ● By Each element divided by the square root of the sum of the squares of the elements in that row
        norms = sqrt.(sum(U.^2, dims=2))
        T = U ./ (norms .+ eps(Float64))  # Add small epsilon to avoid division by zero

        # 5. Cluster rows using k-means
        best_result = nothing
        min_cost = Inf
        for i in 1:5  # Try multiple initializations
            result = kmeans(T', k;          # let yᵢ ∈ ℝᵏ be the vector corresponding to the i-th row of T & k-means algorithm into clusters C₁,...,Cₖ
                        maxiter=200,        # More iterations to ensure convergence
                        display=:none,
                        init=:kmpp,         # Use k-means++ initialization
                        tol=1e-6)           # Tighter convergence
            
            # Totalcost：the sum of the squares of the distances of all points to the center of the cluster to which they belong.
            # The smaller the value, the better the quality of the clusters.
            if result.totalcost < min_cost
                min_cost = result.totalcost
                best_result = result
            end
        end
        return best_result.assignments

    catch e
        if isa(e, ArgumentError)
            rethrow(e)
        else
            @warn "Error in spectral clustering" exception=(e, catch_backtrace())
            throw(ArgumentError("Failed to perform spectral clustering"))
        end
    end
end