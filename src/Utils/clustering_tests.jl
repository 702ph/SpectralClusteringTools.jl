using Random
using Plots
using LinearAlgebra: norm
using .SpectralClusteringTools

export run_clustering_example

const TEST_SEED = 42
const SAVE_DIR = "Utils/results"

# if !isdir(SAVE_DIR)
#     mkdir(SAVE_DIR)
# end

function run_all_shape_tests()
    Random.seed!(TEST_SEED)
    
    run_spirals_clustering_test()
    run_blobs_test()
    run_moons_test()
    run_circles_test()
end

function plot_clustering_comparison(X, njw_results, st_results, nc_results, data_type::String="")
    # Define color scheme for better visualization
    cluster_colors = [
        RGB(0.95, 0.5, 0.5),    # Coral pink
        RGB(0.7, 0.85, 0.9),    # Soft blue
        RGB(0.95, 0.8, 0.2),    # Warm yellow
        RGB(0.6, 0.8, 0.6),     # Sage green
        RGB(0.85, 0.7, 0.9),    # Soft lavender
        RGB(1.0, 0.6, 0.4),     # Soft terracotta
        RGB(0.5, 0.7, 0.9),     # Muted azure
        RGB(0.9, 0.7, 0.5)      # Soft peach
    ]
    
    # Create four larger scatter plots
    p1 = scatter(X[1,:], X[2,:],
        group = fill(1, size(X,2)),
        title = "Original $(data_type) Data",
        xlabel = "X", ylabel = "Y",
        marker = :circle,
        markersize = 5,        # Increased marker size
        markerstrokewidth = 0,
        legend = false,
        aspect_ratio = :equal,
        color = :blue,
        margin = 20Plots.mm,   # Increased margin
        size = (800, 800)      # Larger plot size
    )
    
    p2 = scatter(X[1,:], X[2,:],
        group = njw_results,
        title = "NJW Clustering",
        xlabel = "X", ylabel = "Y",
        marker = :circle,
        markersize = 5,
        markerstrokewidth = 0,
        legend = false,
        aspect_ratio = :equal,
        palette = cluster_colors,
        margin = 20Plots.mm,
        size = (800, 800)
    )
    
    p3 = scatter(X[1,:], X[2,:],
        group = st_results,
        title = "Self-tuning Clustering",
        xlabel = "X", ylabel = "Y",
        marker = :circle,
        markersize = 5,
        markerstrokewidth = 0,
        legend = false,
        aspect_ratio = :equal,
        palette = cluster_colors,
        margin = 20Plots.mm,
        size = (800, 800)
    )
    
    p4 = scatter(X[1,:], X[2,:],
        group = nc_results,
        title = "Normalized Cut Analysis",
        xlabel = "X", ylabel = "Y",
        marker = :circle,
        markersize = 5,
        markerstrokewidth = 0,
        legend = false,
        aspect_ratio = :equal,
        palette = cluster_colors,
        margin = 20Plots.mm,
        size = (800, 800)
    )
    
    # Combine the four plots in a 2x2 grid
    return plot(p1, p2, p3, p4,
        layout = (2,2),
        size = (2000, 2000),   # Much larger overall size
        titlefontsize = 16,    # Larger title font
        plot_titlevspan = 0.1  # Adjust title spacing
    )
end

function plot_affinity_heatmap(W)
    return heatmap(W,
        title = "Normalized Cut Affinity Matrix",
        xlabel = "Point Index",
        ylabel = "Point Index",
        color = :viridis,
        aspect_ratio = :equal,
        margin = 20Plots.mm,
        size = (1000, 800),    # Large size for better detail
        titlefontsize = 14
    )
end

function run_moons_test()
    Random.seed!(TEST_SEED)
    println("\nTesting on Moons:")
    println("===============================")
    X, initial_labels = make_moons()

    njw_params = SpectralClusteringParams(:ε, 15, 0.7, 0.05)
    njw_results = spectral_clustering(X, 3, njw_params)
    
    params = SelfTuningParams(15,true)
    
    predicted_labels, best_C, analysis_info = self_tuning_spectral_clustering(X,2,params)

    nc_params = NormalizedCutsParams(0.1, 0.1, 0.2, 5, 10)
    spatial_coords = Matrix(X')
    features = Matrix(X')
    nc_results, W = normalized_cuts_segmentation(spatial_coords, features, nc_params)

        # Create and save visualizations
    clustering_plot = plot_clustering_comparison(X, njw_results, predicted_labels, nc_results, "Moons")
    affinity_plot = plot_affinity_heatmap(W)
    
    #savefig(clustering_plot, joinpath(SAVE_DIR, "Moons.png"))
    #savefig(affinity_plot, joinpath(SAVE_DIR, "moons_affinity.png"))
    
    # Print performance metrics
    print_efficiency(initial_labels, njw_results)
    print_efficiency(initial_labels, predicted_labels)
    print_efficiency(initial_labels, nc_results)

    display(clustering_plot)
    display(affinity_plot)
    
end


function run_spirals_clustering_test()
    Random.seed!(TEST_SEED)
    println("\nTesting on Spirals:")
    println("===============================")
    # Generate test data with labels
    points_with_labels, true_labels = make_spirals_2d(500, 0.1)
    
    # Only use the points (ignore labels)
    X = Matrix(points_with_labels')
    
    # NJW clustering
    njw_params = SpectralClusteringParams(:ε, 15, 1.5, 0.1)
    njw_results = spectral_clustering(X, 2, njw_params)
    
    # Self-tuning clustering
    st_params = SelfTuningParams(7, true)
    st_results, best_C, _ = self_tuning_spectral_clustering(X, 3, st_params)
    
    # Normalized cuts
    nc_params = NormalizedCutsParams(0.2, 0.2, 0.05, 3, 20)
    spatial_coords = Matrix(X')
    features = Matrix(X')
    nc_results, W = normalized_cuts_segmentation(spatial_coords, features, nc_params)
    
    # Create visualizations
    clustering_plot = plot_clustering_comparison(X, njw_results, st_results, nc_results, "Spirals")
    affinity_plot = plot_affinity_heatmap(W)

    #savefig(clustering_plot, joinpath(SAVE_DIR, "Spirals.png"))
    #savefig(affinity_plot, joinpath(SAVE_DIR, "spirals_affinity.png"))
    
    # Print analysis results
    println("\nSpirals Clustering Analysis Summary:")
    println("Number of clusters discovered:")
    println("  NJW Spectral Clustering: ", length(unique(njw_results)))
    println("  Self-tuning Clustering: ", length(unique(st_results)))
    println("  Normalized Cuts: ", length(unique(nc_results)))
    
    # print_efficiency(initial_labels, njw_results)
    # print_efficiency(initial_labels, predicted_labels)
    # print_efficiency(initial_labels, nc_results)

    display(clustering_plot)
    display(affinity_plot)
end


function run_blobs_test()
    Random.seed!(TEST_SEED)
    println("\nTesting on Blobs:")
    println("===============================")
    # Generate test data with labels
    points_with_labels, true_labels = make_blobs_2d(6, 300, 0.1)
    
    # Only use the points (ignore labels)
    X = Matrix(points_with_labels')
    
    # NJW clustering
    njw_params = SpectralClusteringParams(:ε, 5, 5.0, 0.5)
    njw_results = spectral_clustering(X, 20, njw_params)
    
    # Self-tuning clustering
    st_params = SelfTuningParams(45, true)
    st_results, best_C, _ = self_tuning_spectral_clustering(X, 12, st_params)
    
    # Normalized cuts
    nc_params = NormalizedCutsParams(0.3, 0.1, 0.03, 15, 25)
    spatial_coords = Matrix(X')
    features = Matrix(X')
    nc_results, W = normalized_cuts_segmentation(spatial_coords, features, nc_params)
    
    # Create visualizations
    clustering_plot = plot_clustering_comparison(X, njw_results, st_results, nc_results, "Blobs")
    affinity_plot = plot_affinity_heatmap(W)

    #savefig(clustering_plot, joinpath(SAVE_DIR, "Blobs.png"))
    #savefig(affinity_plot, joinpath(SAVE_DIR, "blobs_affinity.png"))
    
    # Print analysis results
    println("\nBlobs Clustering Analysis Summary:")
    println("Number of clusters discovered:")
    println("  NJW Spectral Clustering: ", length(unique(njw_results)))
    println("  Self-tuning Clustering: ", length(unique(st_results)))
    println("  Normalized Cuts: ", length(unique(nc_results)))
    

    display(clustering_plot)
    display(affinity_plot)
end

function run_circles_test()
    Random.seed!(TEST_SEED)
    println("\nTesting on Circles:")
    println("===============================")
    # Generate test data with labels
    points_with_labels, true_labels = make_circles(3, 500, 0.05, false)
    
    # Only use the points (ignore labels)
    X = Matrix(points_with_labels')
    
    # NJW clustering
    njw_params = SpectralClusteringParams(:knn, 60, 5.0, 2.0)
    njw_results = spectral_clustering(X, 3, njw_params)
    
    # Self-tuning clustering
    st_params = SelfTuningParams(5, true)
    st_results, best_C, _ = self_tuning_spectral_clustering(X, 5, st_params)
    
    # Normalized cuts
    nc_params = NormalizedCutsParams(0.2, 0.2, 0.05, 3, 20)
    spatial_coords = Matrix(X')
    features = Matrix(X')
    nc_results, W = normalized_cuts_segmentation(spatial_coords, features, nc_params)
    
    # Create visualizations
    clustering_plot = plot_clustering_comparison(X, njw_results, st_results, nc_results, "Cirlcles")
    affinity_plot = plot_affinity_heatmap(W)

    #savefig(clustering_plot, joinpath(SAVE_DIR, "Circles.png"))
    #savefig(affinity_plot, joinpath(SAVE_DIR, "circles_affinity.png"))
    
    # Print analysis results
    println("\n Circles Clustering Analysis Summary:")
    println("Number of clusters discovered:")
    println("  NJW Spectral Clustering: ", length(unique(njw_results)))
    println("  Self-tuning Clustering: ", length(unique(st_results)))
    println("  Normalized Cuts: ", length(unique(nc_results)))
    
    # print_efficiency(initial_labels, njw_results)
    # print_efficiency(initial_labels, predicted_labels)
    # print_efficiency(initial_labels, nc_results)

    display(clustering_plot)
    display(affinity_plot)
end

function run_clustering_example(dataset_type::String)
    # Validate input
    valid_types = ["circles", "spirals", "blobs", "moon"]
    dataset_type = lowercase(dataset_type)
    
    if !(dataset_type in valid_types)
        error("Invalid dataset type. Available options: $(join(valid_types, ", "))")
    end
    
    # Call appropriate test function based on dataset type
    if dataset_type == "circles"
        run_circles_test()
    elseif dataset_type == "spirals"
        run_spirals_clustering_test()
    elseif dataset_type == "blobs"
        run_blobs_test()
    elseif dataset_type == "moon"
        run_moons_test()
    end
end

#results = run_clustering_example("moon")
