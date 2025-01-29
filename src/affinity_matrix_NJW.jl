"""
    compute_normalized_laplacian(W::Matrix{Float64})

Computes the normalized Laplacian matrix `Lsym` using the formula:

    ``Lsym = D^(−1/2)LD^(−1/2) = I − D^(-1/2) W D^(-1/2)``

where ``D`` is the degree matrix.

# Arguments
- `W`: A symmetric adjacency matrix (weight matrix) of the graph.

# Returns
- The normalized Laplacian matrix `Lsym`.

# Notes
- If any node has a degree of zero (isolated vertex), small connections are added to ensure numerical stability.
- The resulting Laplacian matrix is symmetrized and clamped to the range `[-1, 1]` to maintain stability.

# Example
```julia
W = [0.0 1.0 0.5; 1.0 0.0 1.0; 0.5 1.0 0.0]
Lsym = compute_normalized_laplacian(W)
```
"""
function compute_normalized_laplacian(W::Matrix{Float64})
    # Compute degree matrix D
    d = sum(W, dims=2)[:]  # Get degrees as a vector
    
    # Check for isolated vertices (degree = 0)
    # Handle isolated vertices by adding small connections to all other vertices
    if any(d .== 0)
        n = size(W, 1)
        eps_val = 1e-10  # Small value for numerical stability
        for i in 1:n
            if d[i] == 0
                # Add small connections to all other vertices
                W[i, :] .= eps_val
                W[:, i] .= eps_val
                W[i, i] = 0  # Keep diagonal as 0 (similarity of data point itself)
            end
        end
        # Recompute degrees
        d = sum(W, dims=2)[:]
    end
    
    
    # Compute D^(-1/2)
    d_inv_sqrt = 1.0 ./ sqrt.(d)
    D_inv_sqrt = Diagonal(d_inv_sqrt) # diagonal matrix
    
    # Compute normalized Laplacian
    L = D_inv_sqrt * W * D_inv_sqrt
    
    # Ensure numerical stability
    L = (L + L') / 2  # Make sure L is symmetric
    L = clamp.(L, -1.0, 1.0)  # Clamp values to [-1,1]
    
    return L
end