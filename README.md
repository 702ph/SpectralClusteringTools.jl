# SpectralClusteringTools

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://702ph.github.io/SpectralClusteringTools.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://702ph.github.io/SpectralClusteringTools.jl/dev/)
[![Build Status](https://github.com/702ph/SpectralClusteringTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/702ph/SpectralClusteringTools.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/702ph/SpectralClusteringTools.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/702ph/SpectralClusteringTools.jl)


# SpectralClusteringTools

*Tools for Spectral Clustering.*

A package for creating test data and providing functions to facilitate clustering analysis.


## Package Features
- Tools for generating synthetic test datasets for clustering experiments.
- Functions for preprocessing data to prepare for spectral clustering.
- Implementation of spectral clustering algorithms.
- Utilities for visualizing clustering results and data distributions.


## Installation
This guide will help you set up and install the SpectralClusteringTools.jl package, which is developed as part of a university project for Julia version 1.10. Follow the steps below to ensure a successful installation.

### Step 1: Cloning the Git Repository
1. Open a terminal (command line interface) on your system.
2. Run the following commands to clone the repository and navigate to its directory:
```
git clone "https://github.com/702ph/SpectralClusteringTools.jl"
cd SpectralClusteringTools.jl
git checkout develop
git pull origin

```
- The first command downloads the repository from GitHub to your local machine.
- The second command changes your current directory to the repository folder, where the package is located.


3. After navigating to the project folder, launch Julia by typing:
```
julia
```
This starts the Julia REPL (Read-Eval-Print Loop), an interactive environment where you can execute Julia commands.


### Step 2: Getting the Package Ready for Use
Once the Julia REPL has started, you should perform the necessary steps to use the package. Follow these steps

1. Inside the Julia REPL, import the package manager by typing:
```
using Pkg
```


2. Activate the local environment for the project by running:
```
Pkg.activate(".")
```
- The `activate()` command tells Julia to use the package environment defined in the current directory (`.` refers to the current folder where the project was cloned).

3. Install all dependencies specified in the project's Project.toml file:
```
Pkg.instantiate()
```
The `instantiate()` command ensures all required packages for the project are downloaded and installed.



### Notes and Troubleshooting
- *Julia Version*: This project is developed and tested with *Julia version 1.10*. Ensure that you have this version installed on your system. You can check your Julia version by running:
```
julia --version
```
- *Dependencies*: The Pkg.instantiate() command will automatically install all required dependencies. If you encounter issues, try running Pkg.update() to ensure you have the latest compatible versions of the packages.


## Example
In this section, we will demonstrate how to generate and cluster synthetic datasets using different data generation functions. Follow the step-by-step instructions below.

### Step 1: Load Required Packages
First, load the necessary Julia packages:

```julia
using SpectralClusteringTools
using LinearAlgebra
using Plots
using Random
using Statistics
```


If the compilation of Plots fails, you may need to roll back the version of GR to 0.73.10 by running the following command. For more details on this issue, refer to the following link: https://github.com/jheinen/GR.jl/issues/556

```julia
Pkg.add(PackageSpec(name="GR", version="0.73.10"))
```

Next, to enable interactive operations, set *Plotly* as the backend for *Plots*:
```julia
import PlotlyJS
plotlyjs()
```

### Step 2: Define a Visualization Function
Define helper functions (These function is planned to be included in the package in the future.)
```julia
# visualize given data as a 3D scatter plot grouped by labels.
function visualize_3d_clusters(points, labels, title; legendpos=:topright)
    p = scatter(
        points[:, 1], points[:, 2], points[:, 3],
        group = labels,
        title = title,
        xlabel = "X", ylabel = "Y", zlabel = "Z",
        aspect_ratio = :auto,
        marker = :circle,
        markersize = 0.5,
        alpha = 0.7,
        camera = (45, 45),
        grid = false,
        background_color = :white,
        legend = legendpos
    )
    return p
end



# Computing the best sigma
function calculate_optimal_sigma(X)
    n_points = size(X, 2)
    k_nearest = min(7, n_points - 1)
    dists = zeros(n_points)

    for i in 1:n_points
        current_point_dists = Float64[]
        for j in 1:n_points
            if i != j
                distance = norm(X[:,i] - X[:,j])
                push!(current_point_dists, distance)
            end
        end
        sorted_dists = sort(current_point_dists)
        if length(sorted_dists) >= k_nearest
            dists[i] = mean(sorted_dists[1:k_nearest])
        else
            dists[i] = mean(sorted_dists)
        end
    end
    
    local_sigma = median(dists)
    println("Dist Summ")
    println("  Min dist", round(minimum(dists), digits=4))
    println("  Max Dist", round(maximum(dists), digits=4))
    println("  Median Dist", round(local_sigma, digits=4))
    
    optimal_sigma = local_sigma * 0.15
    println("  Computed Sigma:", round(optimal_sigma, digits=4))
    
    return optimal_sigma
end
```


### Step 3: Generate Test Data
Follow the steps below to generate test data, perform clustering, and compare the test data with the clustering results.
```julia
# Test Data Generation Settings 
Random.seed!(42)
num_classes = 3
points_per_class = 100
noise = 0.1
adjust_scale = true
```

We can generate different types of datasets using the available functions:

#### Example 1: Spheres Dataset
The `make_spheres` function generates points distributed on the surfaces of spheres.
```julia
 X, true_labels = make_spheres(num_classes, points_per_class, noise, adjust_scale)
```


#### Example 2: Lines Dataset
The `make_lines` function generates points distributed along straight lines.
```julia
X, true_labels = make_lines(num_classes, points_per_class, noise)
```


#### Example 3: Spirals Dataset
The `make_spirals` function generates two interlacing spiral patterns.
```julia
X, true_labels = make_spirals(points_per_class, noise)
```

#### Example 4: Blobs Dataset
The `make_blobs` function generates clusters of points grouped into separate blobs.
```julia
X, true_labels = make_blobs(num_classes, points_per_class, noise)
```

### Step 4: Perform Clustering
Once the data is generated, we can apply spectral clustering to identify clusters within the data.

```julia
# Clustering Preparation
X_norm = (X .- mean(X, dims=1)) ./ std(X, dims=1)
X_clustering = Matrix(X_norm')
sigma = calculate_optimal_sigma(X_clustering)


# Perform Clustering 
params = SpectralClusteringParams(
:fully_connected, 7, 7.0, sigma)
predicted_labels = spectral_clustering(X_clustering, num_classes, params)


# Visualize the Result
# (Make sure to add a semicolon (;) at the end, otherwise the graph sub-window will open.)
p1 = visualize_3d_clusters(X, true_labels, "Original Spherical Data", legendpos=:bottomleft);
p2 = visualize_3d_clusters(X, predicted_labels, "Spectral Clustering Results");
plot(p1, p2, layout=(1,2), size=(1200,600))
```



## Documentation
The [documentation](https://702ph.github.io/SpectralClusteringTools.jl/dev/) of the most recently version.