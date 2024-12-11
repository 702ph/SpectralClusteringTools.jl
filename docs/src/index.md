```@meta
CurrentModule = SpectralClusteringTools
```

# SpectralClusteringTools

*Tools for Spectral Clustering.*

A package for creating test data and providing functions to facilitate clustering analysis.


## Package Features
- Tools for generating synthetic test datasets for clustering experiments.
- Functions for preprocessing data to prepare for spectral clustering.
- Implementation of spectral clustering algorithms.
- Utilities for visualizing clustering results and data distributions.


## Getting Started
### Installation

Install using the Julia's package manager. In the Julia REPL, type `] for entering the Pkg REPL mode and run

```
pkg> add SpectralClusteringTools
```
type CTRL + C to back the Julia REPL. Equivalently with the Pkg API:
```
julia> import Pkg 
Julia> Pkg.add("SpectralClusteringTools")
```


## Example
```
using SpectralClusteringTools
using Plots
import PlotlyJS


# Use Plotly as the backend for Plots.jl
plotlyjs()

# A function for visualize given data as a 3D scatter plot grouped by labels.
function plot_data(data, labels, title="Visualization")
    scatter3d(data[:, 1], data[:, 2], data[:, 3],
              group=labels, title=title,
              xlabel="X", ylabel="Y", zlabel="Z", aspect_ratio=:auto,
              markersize=0.5)
end


# Generate test data
sphere_data, sphere_labels = make_sphere(3, 200)


# Visualize data
plot_data(sphere_data, sphere_labels, "Multiclass Spheres(Original)")


# Clustering 
#TODO: add or fixcode here
graph = epsilon_neighborhood_graph(sphere_data)
???? = next_fuction(graph)

clustered_labels = ????(???)


# Visualize
#TODO: add or fix code here
plot_data(sphere_data, clustered_labels, "Multiclass Spheres(Clustered)")

```



## Documentation 
Documentation for [SpectralClusteringTools](https://github.com/702ph/SpectralClusteringTools.jl)'s interface.




```@index
```


```@autodocs
Modules = [SpectralClusteringTools]
```
