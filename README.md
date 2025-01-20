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
In this section, we will explain step by step how to perform clustering on data. First, load the required packages:
```
using SpectralClusteringTools
using Random
using Plots
```


If the compilation of Plots fails, you may need to roll back the version of GR to 0.73.10 by running the following command. For more details on this issue, refer to the following link: https://github.com/jheinen/GR.jl/issues/556

```
Pkg.add(PackageSpec(name="GR", version="0.73.10"))
```

Next, to enable interactive operations, set *Plotly* as the backend for *Plots*:
```
import PlotlyJS
plotlyjs()
```

Now, let's create a function to visualize the data. (This function is planned to be included in the package in the future.)
```
# A function for visualize given data as a 3D scatter plot grouped by labels.
function plot_data(data, labels, title="Visualization")
    scatter3d(data[:, 1], data[:, 2], data[:, 3],
              group=labels, title=title,
              xlabel="X", ylabel="Y", zlabel="Z", aspect_ratio=:auto,
              markersize=0.5)
end
```


Follow the steps below to generate test data, perform clustering, and compare the test data with the clustering results.
```
<!-- # Generate Test Data 
points, labels = make_spheres(3, 200, 0.1, true) -->

# Test Data Generation Settings 
Random.seed!(42)
num_classes = 3
points_per_class = 100
noise = 0.1
adjust_scale = True

# Generate Test Data 
 X, true_labels = make_spheres(num_classes, points_per_class, noise, adjust_scale)

<!-- # Visualize data
plot_data(points, labels, "Multiclass Spheres(Original)") -->



# Perfoming Clustering 
#TODO: add or fixcode here
graph = epsilon_neighborhood_graph(sphere_data)
???? = next_fuction(graph)

clustered_labels = ????(???)


# Visualize the result
#TODO: add or fix code here
plot_data(sphere_data, clustered_labels, "Multiclass Spheres(Clustered)")

```



## Documentation
- [**STABLE**][docs-stable-url] &mdash; **documentation of the most recently tagged version.**
- [**DEVEL**][docs-dev-url] &mdash; *documentation of the in-development version.*
