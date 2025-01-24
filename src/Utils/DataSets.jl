export make_spheres
export make_lines
export make_spirals
export make_blobs

"""
    make_spheres(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0, adjust_scale::Bool=true)

Generates a dataset of points distributed on the surfaces of spheres. 
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

"""
    make_lines(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)

Generates a dataset of points distributed following straight lines.
Each line corresponds to a class.

# Arguments
- `num_classes`: Number of lines (classes) to generate.
- `num_points_per_class`: Number of points per class.
- `noise`: Maximal noise applied for each point.

# Returns
- A tuple of `(normalized_points, labels)`, where `normalized_points` is a matrix of 3D points, and `labels` indicates class membership.

# Example
`points, labels = make_lines(5, 500, 0.3)`
"""	
function make_lines(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)
	points = Matrix{Float64}(undef, 0, 3)
	labels = Vector{Int}(undef, 0)

	for class in 1:num_classes
        # Generate points
		x = rand(num_points_per_class)
		y = rand(num_points_per_class)
		z = [class for _ in 1:num_points_per_class]

        # Add noise
		noise_values = noise .* (2 .* rand(num_points_per_class) .- 1)
		z = z + noise_values

        # Append points and labels for the current class
		points_for_class = hcat(x, y, z)
        points = vcat(points, points_for_class)
        labels = vcat(labels, fill(class, num_points_per_class))

	end

    # Normalized points so that they lie between 0 and 1
	normalized_points = (points .- minimum(points)) ./ (maximum(points) -minimum(points))
    return normalized_points, labels

end

"""
    make_spirals(num_points_per_class::Int, noise::Float64=0.0)

Generates a dataset of points following 2 interlacing spirals.
Each spiral corresponds to a class.

# Arguments
- `num_points_per_class`: Number of points per class.
- `noise`: Maximal noise applied for each point.

# Returns
- A tuple of `(normalized_points, labels)`, where `normalized_points` is a matrix of 3D points, and `labels` indicates class membership.

# Example
`points, labels = make_spirals(500, 1.0)`
"""	
function make_spirals(num_points_per_class::Int, noise::Float64=0.0)
	points = Matrix{Float64}(undef, 0, 3)
	labels = Vector{Int}(undef, 0)

    # Generate a spiral
	function generate_spiral(num_points_per_class::Int, noise::Float64=0.0)
        # Generation of angular distance
		n = sqrt.(rand(num_points_per_class)) .* 720 .* (2π / 360)

        # Calculation of x, y, z coordinates with noise added
	    d1x = -cos.(n) .* n + rand(num_points_per_class) .* noise
	    d1y = sin.(n) .* n + rand(num_points_per_class) .* noise
		d1z = n .+ rand(num_points_per_class) .* noise

		return hcat(d1x, d1y, d1z)
	end

    # Generate both spirals
	spiral1 = generate_spiral(num_points_per_class, noise)
	spiral2 = generate_spiral(num_points_per_class, noise) .+ 3
	
    # Append points and labels
	points = vcat(spiral1, spiral2)
	labels = vcat([1 for _ in 1:num_points_per_class],[2 for _ in 1:num_points_per_class])

    return points, labels

end

"""
    make_blobs(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)

Generates a dataset of points distributed on different blobs. 
Each blob corresponds to a class.

# Arguments
- `num_classes`: Number of spheres (classes) to generate.
- `num_points_per_class`: Number of points per class.
- `noise`: Maximal noise applied for each point.

# Returns
- A tuple of `(normalized_points, labels)`, where `normalized_points` is a matrix of 3D points, and `labels` indicates class membership.

# Example
`points, labels = make_blobs(5, 500, 0.5)`
"""	
function make_blobs(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)
	points = Matrix{Float64}(undef, 0, 3)
	labels = Vector{Int}(undef, 0)

	for class in 1:num_classes
        # Generate noise
		noise_values = noise .* (2 .* rand(num_points_per_class) .- 1)
		
        # Generate points with noise
		x = rand(num_points_per_class) + noise_values .+ 2*class
		y = rand(num_points_per_class) + noise_values .+ 2*class
		z = rand(num_points_per_class) + noise_values .+ 2*class

        # Append points and labels for the current class
		points_for_class = hcat(x, y, z)
        points = vcat(points, points_for_class)
        labels = vcat(labels, fill(class, num_points_per_class))

	end
	
	return points, labels
end