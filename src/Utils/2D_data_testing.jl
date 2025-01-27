using .SpectralClusteringTools
using Plots
export generate_mixed_concentric_data, generate_mixed_moons_data
"""
generate_mixed_concentric_data()

Generate a dataset of mixed concentric circles.

This function creates a dataset with three main concentric circles, each with two smaller concentric circles mixed in. The data points are generated with a small amount of random noise to make the task more challenging.

Returns:
- `X::Matrix{Float64}`: The matrix of data points, with each column representing a data point.
- `labels::Vector{Int}`: The vector of true cluster labels for each data point.
"""
function generate_mixed_concentric_data()
    n_points = 500 
    
    function generate_circle(r::Float64, n::Int, noise_level::Float64=0.1)
        θ = range(0, 2π, length=n)
        noise_r = randn(n) .* noise_level
        return [(r .+ noise_r) .* cos.(θ) (r .+ noise_r) .* sin.(θ)]
    end
    
    r1_main = generate_circle(1.0, n_points)
    r1_mix1 = generate_circle(2.0, n_points÷5)  
    r1_mix2 = generate_circle(3.0, n_points÷8)  
    
    r2_main = generate_circle(2.0, n_points)
    r2_mix1 = generate_circle(1.0, n_points÷6)
    r2_mix2 = generate_circle(3.0, n_points÷6)
    
    r3_main = generate_circle(3.0, n_points)
    r3_mix1 = generate_circle(1.0, n_points÷8)
    r3_mix2 = generate_circle(2.0, n_points÷5)
    
    X1 = vcat(r1_main, r1_mix1, r1_mix2)
    labels1 = fill(1, size(X1, 1))  # All points in X1 are labeled as 1
    
    X2 = vcat(r2_main, r2_mix1, r2_mix2)
    labels2 = fill(1, size(X2, 1))  # All points in X2 are labeled as 1
    
    X3 = vcat(r3_main, r3_mix1, r3_mix2)
    labels3 = fill(1, size(X3, 1))  # All points in X3 are labeled as 1

    X = vcat(X1, X2, X3)
    labels = vcat(labels1, labels2, labels3)  # All points are labeled as 1
    
    return Matrix(X'), labels 
end

"""
generate_mixed_moons_data(n_samples::Int=300, noise::Float64=0.05)

    Generate two interleaved moon-shaped datasets with overlapping regions. This function is primarily used for creating synthetic data to test clustering algorithms.

    # Arguments
    - `n_samples::Int=300`: Number of base samples for each moon shape
    - `noise::Float64=0.05`: Standard deviation of Gaussian noise added to the data points

    # Returns
    Returns a tuple `(X, labels)`:
    - `X`: 2×N matrix where N = 2 * (n_samples + n_samples÷3), each column represents a 2D data point
    - `labels`: Vector of length N containing class labels (1 or 2) for each point

    # Implementation Details
    1. Generates base data points for two moon shapes
    2. Adds Gaussian noise to each data point
    3. Generates additional mixing points (n_samples÷3 points) for each class
    4. Mixing points have higher noise (2x the base noise)

    # Example
    ```julia
    X, labels = generate_mixed_moons_data(500, 0.1)
    scatter(X[1,:], X[2,:], group=labels, title="Mixed Moons Dataset")
"""

function generate_mixed_moons_data(n_samples::Int=300, noise::Float64=0.05)
    
    t = range(0, π, length=n_samples)
    x1 = cos.(t)
    y1 = sin.(t)

    x2 = 0.5 .- cos.(t) 
    y2 = 0.3 .- sin.(t)  
    
    x1 .+= randn(length(x1)) .* noise
    y1 .+= randn(length(y1)) .* noise
    x2 .+= randn(length(x2)) .* noise
    y2 .+= randn(length(y2)) .* noise
    
    n_mix = n_samples ÷ 3  
    
    mix_indices1 = rand(1:n_samples, n_mix)
    mix_x1 = x1[mix_indices1] .+ randn(n_mix) .* noise * 2
    mix_y1 = y1[mix_indices1] .+ randn(n_mix) .* noise * 2
    
    mix_indices2 = rand(1:n_samples, n_mix)
    mix_x2 = x2[mix_indices2] .+ randn(n_mix) .* noise * 2
    mix_y2 = y2[mix_indices2] .+ randn(n_mix) .* noise * 2
    
    X1 = hcat(x1, y1)
    X2 = hcat(x2, y2)
    Mix1 = hcat(mix_x2, mix_y2)
    Mix2 = hcat(mix_x1, mix_y1)
    
    X = vcat(X1, Mix1, X2, Mix2)
    labels = fill(1, size(X, 1))  # All points are labeled as 1
    
    return Matrix(X'), labels
end
