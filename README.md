# SpectralClusteringTools

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://702ph.github.io/SpectralClusteringTools.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://702ph.github.io/SpectralClusteringTools.jl/dev/)
[![Build Status](https://github.com/702ph/SpectralClusteringTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/702ph/SpectralClusteringTools.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/702ph/SpectralClusteringTools.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/702ph/SpectralClusteringTools.jl)


# SpectralClusteringTools

*Tools for Spectral Clustering.*

A package for creating test data and providing functions to facilitate clustering analysis. The implemented methods follow the approach outlined in A Tutorial on Spectral Clustering by Ulrike von Luxburg (2007). Additionally, the self-tuning spectral clustering technique is based on the methodology described in Self-Tuning Spectral Clustering by Zelnik-Manor & Perona (2004). This allows users to experiment with different clustering scenarios and effectively visualize results.


## Package Features
- Tools for generating synthetic test datasets for clustering experiments.
- Functions for preprocessing data to prepare for spectral clustering.
- Implementation of spectral clustering algorithms.
- Utilities for visualizing clustering results and data distributions.


## Prerequisites

| Prerequisite | Version | Installation Guide | Required |
|--------------|---------|--------------------|----------|
| Julia       | 1.10    | [![Julia](https://img.shields.io/badge/Julia-v1.10-blue)](https://julialang.org/downloads/) | ✅ |



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
```julia
using Pkg
```


2. Activate the local environment for the project by running:
```julia
Pkg.activate(".")
```
- The `activate()` command tells Julia to use the package environment defined in the current directory (`.` refers to the current folder where the project was cloned).

3. Install all dependencies specified in the project's Project.toml file:
```julia
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

### Quick Start: Generate and Cluster Data in One Step

To simplify the process for users, we have consolidated data generation, clustering, and visualization into a single function. This allows you to execute the entire workflow with just one command.

#### Step 1: Load Required Packages

Before running the function, ensure that you have the necessary packages loaded:

```julia
using SpectralClusteringTools
using LinearAlgebra
using Plots
using Random
using Statistics
```

#### Step 2: Run the Clustering Pipeline

Now, simply call the `run_clustering_example` function to generate test data, apply spectral clustering, and visualize the results:

```julia
# Available options: "circles", "spirals", "blobs", "moon"
run_clustering_example("circles")  
```

Each dataset type represents a different clustering scenario:

- **"circles"**: Generates concentric circles of points, useful for testing algorithms that handle nested structures.
- **"spirals"**: Produces two interlacing spiral patterns, commonly used for evaluating algorithms that handle non-linear separability.
- **"blobs"**: Generates clusters of points arranged in distinct groups with a shifting center for each class, rather than following a Gaussian distribution.
- **"moon"**: Creates two crescent-shaped clusters, which are often used for assessing clustering methods on non-linearly separable data.

This function performs the following steps automatically:

- Generates a synthetic dataset based on the selected type.
- Normalizes the data for clustering.
- Computes the optimal sigma for affinity matrix calculation.
- Applies spectral clustering to identify clusters.
- Visualizes both the original data and the clustering results.

#### Example Output

After running the function, you will see two side-by-side 3D scatter plots:

1. **Original Data Distribution** – Displays the true class labels.
2. **Spectral Clustering Results** – Shows the predicted clusters.

You can experiment with different dataset types by changing the argument in `run_clustering_example("dataset_type")`.


#### Notes & Troubleshooting
- Ensure you are using **Julia 1.10**.
- If the compilation of Plots fails, you may need to roll back the version of GR to 0.73.10 by running the following command. For more details on this issue, refer to the following link: https://github.com/jheinen/GR.jl/issues/556

  ```julia
  Pkg.add(PackageSpec(name="GR", version="0.73.10"))
  ```
- If necessary, manually update dependencies using:
  ```julia
  Pkg.update()
  ```



## Documentation
For more detailed explanations and advanced usage, 
please refer to the [documentation](https://702ph.github.io/SpectralClusteringTools.jl/stable/). 
With this streamlined approach, users can quickly experiment 
with spectral clustering on different dataset types with minimal effort!


## References
- **A Tutorial on Spectral Clustering**, Ulrike von Luxburg, 2007 [Link](https://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf)
- **Self-Tuning Spectral Clustering**, Zelnik-Manor & Perona, 2004 [Link](https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf)



## Contributors
[![Contributors](https://contrib.rocks/image?repo=702ph/SpectralClusteringTools.jl)](https://github.com/702ph/SpectralClusteringTools.jl/graphs/contributors)


