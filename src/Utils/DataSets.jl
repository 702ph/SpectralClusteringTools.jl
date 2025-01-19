export make_sphere


"""
make_sphere(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0, adjust_scale::Bool=true)
Generates a dataset of points distributed on the surfaces of spheres。 
Each sphere corresponds to a class, and points are normalized.

# Arguments
- `num_classes`: Number of spheres (classes) to generate.
- `num_points_per_class`: Number of points per class.
- `noise`: Maximal noise applied for each point.
- `adjust_scale`: Adjusts the number of points.

# Returns
- A tuple of `(normalized_points, labels)`, where `normalized_points` is a matrix of 3D points, and `labels` indicates class membership.

# Example
`points, labels = make_sphere(3, 1000, 0.1, false)`
"""	
function make_sphere(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0, adjust_scale::Bool=true)
	points = Matrix{Float64}(undef, 0, 3)
	labels = Vector{Int}(undef, 0)                	
	    
	for class in 1:num_classes
		# Radius of the current class
        radius = class

		# Scale w.r.t. area of first sphere (4π)
		scale = adjust_scale ? (4π * radius^2) / 4π : 1
			
        # Generate points in spherical coordinate
        num_points = Int(round(num_points_per_class * scale))
        θ = 2π .* rand(num_points)  # Azimuthal angle (0 to 2π)
        cos_ϕ = 2 .* rand(num_points) .- 1  # Cosine of polar angle uniform in [-1,1]
        ϕ = acos.(cos_ϕ)                    # Polar angle from cosine

		# Add noise to the radius
        noise_values = noise .* (2 .* rand(num_points) .- 1)
        perturbed_radius = radius .+ noise_values
		
		# conver to cartesian
        x = perturbed_radius .* cos.(θ) .* sin.(ϕ)
        y = perturbed_radius .* sin.(θ) .* sin.(ϕ)
        z = perturbed_radius .* cos_ϕ

		# Append points and labels for the current class
        points_for_class = hcat(x, y, z)
        points = vcat(points, points_for_class)
        labels = vcat(labels, fill(class, num_points))
    end

    normalized_points = (points .- minimum(points)) ./ (maximum(points) - minimum(points))
    return normalized_points, labels
end