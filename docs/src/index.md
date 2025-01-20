```@meta
CurrentModule = SpectralClusteringTools
```

# SpectralClusteringTools


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
sphere_data, sphere_labels = make_spheres(3, 200)


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
