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

result = run_mixed_circle_test()