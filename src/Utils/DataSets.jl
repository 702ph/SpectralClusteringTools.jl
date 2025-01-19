export make_sphere


"""
    make_sphere(num_classes::Int, num_points_per_class::Int; adjust_scale::Bool=true)

Generates a dataset of points distributed on the surfaces of spheres。 
Each sphere corresponds to a class, and points are normalized.

# Arguments
- `num_classes`: Number of spheres (classes) to generate.
- `num_points_per_class`:  Number of points per class.
- `adjust_scale`: Adjusts the number of points.

# Returns
- A tuple of `(normalized_points, labels)`, where `normalized_points` is a matrix of 3D points, and `labels` indicates class membership.

# Example
`points, labels = make_sphere(3, 1000, false)`
"""	
function make_sphere(num_classes::Int, num_points_per_class::Int, adjust_scale::Bool=true)
	# Input validation
    num_classes > 0 || throw(ArgumentError("Number of classes must be greater than 0"))
    num_points_per_class > 0 || throw(ArgumentError("Number of points per class must be greater than 0"))
    
    points = Matrix{Float64}(undef, 0, 3)
	labels = Vector{Int}(undef, 0)                	
	    
	for class in 1:num_classes
		# Radius of the current class
        radius = class

		# Scale w.r.t. area of first sphere (4π)
		scale = adjust_scale ? (4π * radius^2) / 4π : 1
			
        # Generate points in spherical coordinate
        num_points = Int(round(num_points_per_class * scale))
        φ = π * rand(num_points)  # azimuthal angles
        θ = 2π * rand(num_points) # polar angles

		# conver to cartesian
        x = radius .* sin.(φ) .* cos.(θ)
        y = radius .* sin.(φ) .* sin.(θ)
        z = radius .* cos.(φ)

		# Append points and labels for the current class
        points_for_class = hcat(x, y, z)
        points = vcat(points, points_for_class)
        labels = vcat(labels, fill(class, num_points))
    end

    normalized_points = (points .- minimum(points)) ./ (maximum(points) - minimum(points))
    return normalized_points, labels
end