export plot_data

using Plots
import PlotlyJS


function __init__()
    plotlyjs()
end

"""
Plot given data as a 3D scatter plot grouped by labels.
(only for develpement period)
"""
function plot_data(data, labels, title="Visualization")
    scatter3d(data[:, 1], data[:, 2], data[:, 3],
              group=labels, title=title,
              xlabel="X", ylabel="Y", zlabel="Z", aspect_ratio=:auto,
              markersize=0.5)
end