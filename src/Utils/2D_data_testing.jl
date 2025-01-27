using .SpectralClusteringTools
using Plots
"""
generate_mixed_concentric_data()

Generate a dataset of mixed concentric circles.

This function creates a dataset with three main concentric circles, each with two smaller concentric circles mixed in. The data points are generated with a small amount of random noise to make the task more challenging.

Returns:
- `X::Matrix{Float64}`: The matrix of data points, with each column representing a data point.
- `labels::Vector{Int}`: The vector of true cluster labels for each data point.
"""
function generate_mixed_concentric_data()
    n_points = 200  
    
    function generate_circle(r::Float64, n::Int, noise_level::Float64=0.05)
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
    labels1 = fill(1, size(X1, 1))
    
    X2 = vcat(r2_main, r2_mix1, r2_mix2)
    labels2 = fill(2, size(X2, 1))
    
    X3 = vcat(r3_main, r3_mix1, r3_mix2)
    labels3 = fill(3, size(X3, 1))

    X = vcat(X1, X2, X3)
    labels = vcat(labels1, labels2, labels3)
    
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
    labels = vcat(
        fill(1, n_samples),
        fill(1, n_mix),
        fill(2, n_samples),
        fill(2, n_mix)
    )
    
    return Matrix(X'), labels
end

"""
run_mixed_circle_test()

Run the self-tuning spectral clustering algorithm on the mixed concentric circles dataset and visualize the results.

This function first generates the mixed concentric circles dataset using the `generate_mixed_concentric_data()` function. It then applies the self-tuning spectral clustering algorithm, using the `self_tuning_spectral_clustering()` function, to cluster the data.

The function returns a plot that displays the original data points colored by their true labels, and the data points colored by the predicted labels from the clustering algorithm.

Returns:
- `result::Any`: The plot object containing the visualization of the original data and the clustering results.
"""
function run_mixed_circle_test()
    X, initial_labels = generate_mixed_concentric_data()
    
    params = SelfTuningParams(
        7,   
        true  
    )
    
    predicted_labels, best_C, analysis_info = self_tuning_spectral_clustering(
        X,
        5,    
        params
    )
    

    p1 = scatter(X[1,:], X[2,:],
    group=initial_labels,
    title="Original Mixed Data",
    xlabel="X", ylabel="Y",
    marker=:circle,
    markersize=3,
    legend=true,
    aspect_ratio=:equal,
    palette=:Set1,  
    seriescolor=:auto  
    )

    
    p2 = scatter(X[1,:], X[2,:],
    group=predicted_labels,
    title="Clustering Results",
    xlabel="X", ylabel="Y",
    marker=:circle,
    markersize=3,
    legend=true,
    aspect_ratio=:equal,
    palette=:Set1
    )
    
    return plot(p1, p2, layout=(1,2), size=(1200,600))
end

"""
run_moons_test()

    Run a test of spectral clustering algorithm on the moon-shaped synthetic dataset.

    # Description
    This function performs the following steps:
    1. Generates synthetic moon-shaped data with mixing regions
    2. Applies self-tuning spectral clustering to the dataset
    3. Visualizes both the original data and clustering results side by side

    # Implementation Details
    - Uses `generate_mixed_moons_data()` to create test data
    - Applies self-tuning spectral clustering with:
    * 15 nearest neighbors
    * Local scaling enabled
    * 2 target clusters
    - Creates two scatter plots:
    * Left: Original data with true labels
    * Right: Data with predicted cluster assignments

    # Returns
    Returns a composed plot with two subplots showing the original data and clustering results
    (plot size: 1200×600 pixels)

    # Example
    ```julia
    # Run the test and display results
    plot = run_moons_test()
    display(plot)
"""
function run_moons_test()
    X, initial_labels = generate_mixed_moons_data()
    
    params = SelfTuningParams(
        15,   
        true  
    )
    
    predicted_labels, best_C, analysis_info = self_tuning_spectral_clustering(
        X,
        2,    
        params
    )
    
    p1 = scatter(X[1,:], X[2,:],
        group=initial_labels,
        title="Original Moons Data",
        xlabel="X", ylabel="Y",
        marker=:circle,
        markersize=3,
        legend=true,
        aspect_ratio=:equal,
        palette=:Set1,
        seriescolor=:auto
    )
    
    p2 = scatter(X[1,:], X[2,:],
        group=predicted_labels,
        title="Clustering Results",
        xlabel="X", ylabel="Y",
        marker=:circle,
        markersize=3,
        legend=true,
        aspect_ratio=:equal,
        palette=:Set1
    )
    
    return plot(p1, p2, layout=(1,2), size=(1200,600))
end


#result = run_mixed_circle_test()
result = run_moons_test()