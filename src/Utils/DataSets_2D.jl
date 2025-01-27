export make_circles
export make_lines_2d
export make_spirals_2d
export make_blobs_2d

"""
    make_circles(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0, adjust_scale::Bool=true)

Generates a dataset of points distributed on concentric circles. 
Each circle corresponds to a class, and points are normalized.

# Arguments
- `num_classes`: Number of circles (classes) to generate.
- `num_points_per_class`: Number of points per class.
- `noise`: Maximum noise applied to each point.
- `adjust_scale`: Adjusts the number of points per circle based on its circumference.

# Returns
- A tuple of `(normalized_points, labels)`, where `normalized_points` is a matrix of 2D points, and `labels` indicates class membership.

# Example
`points, labels = make_circles(3, 1000, 0.1, false)`
"""
function make_circles(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0, adjust_scale::Bool=true)
    points = Matrix{Float64}(undef, 0, 2)
    labels = Vector{Int}(undef, 0)

    for class in 1:num_classes
        # Radius of the current class
        radius = class

        # Scale relative to the circumference of the first circle
        scale = adjust_scale ? (2π * radius) / (2π) : 1

        # Generate points in polar coordinates
        num_points = Int(round(num_points_per_class * scale))
        θ = 2π .* rand(num_points)  # Angle (0 to 2π)

        # Add noise to the radius
        noise_values = noise .* (2 .* rand(num_points) .- 1)
        perturbed_radius = radius .+ noise_values

        # Convert to Cartesian coordinates
        x = perturbed_radius .* cos.(θ)
        y = perturbed_radius .* sin.(θ)

        # Append points and labels for the current class
        points_for_class = hcat(x, y)
        points = vcat(points, points_for_class)
        labels = vcat(labels, fill(class, num_points))
    end

    normalized_points = (points .- minimum(points)) ./ (maximum(points) - minimum(points))
    return normalized_points, labels
end

"""
    make_lines_2d(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)

Generates a dataset of points distributed along straight lines in 2D.
Each line corresponds to a class.

# Arguments
- `num_classes`: Number of lines (classes) to generate.
- `num_points_per_class`: Number of points per class.
- `noise`: Maximum noise applied to each point.

# Returns
- A tuple of `(points, labels)`, where `points` is a matrix of 2D points, and `labels` indicates class membership.

# Example
`points, labels = make_lines_2d(5, 500, 0.3)`
"""
function make_lines_2d(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)
    points = Matrix{Float64}(undef, 0, 2)
    labels = Vector{Int}(undef, 0)

    for class in 1:num_classes
        # Generate x coordinates
        x = rand(num_points_per_class)

        # y coordinate is fixed for each class, with added noise
        y = [class for _ in 1:num_points_per_class]
        noise_values = noise .* (2 .* rand(num_points_per_class) .- 1)
        y = y .+ noise_values

        # Combine x and y
        points_for_class = hcat(x, y)
        points = vcat(points, points_for_class)
        labels = vcat(labels, fill(class, num_points_per_class))
    end

    return points, labels
end

"""
    make_spirals_2d(num_points_per_class::Int, noise::Float64=0.0)

Generates a dataset of points forming two interlacing spirals in 2D.
Each spiral corresponds to a class.

# Arguments
- `num_points_per_class`: Number of points per class.
- `noise`: Maximum noise applied to each point.

# Returns
- A tuple of `(points, labels)`, where `points` is a matrix of 2D points, and `labels` indicates class membership.

# Example
`points, labels = make_spirals_2d(500, 0.1)`
"""
function make_spirals_2d(num_points_per_class::Int, noise::Float64=0.0)
    points = Matrix{Float64}(undef, 0, 2)
    labels = Vector{Int}(undef, 0)

    function generate_spiral(num_points_per_class::Int, direction::Int, noise::Float64=0.0)
        θ = sqrt.(rand(num_points_per_class)) .* 4π  # Increasing angle (0 to 4π)
        r = θ  # Radius proportional to the angle
        x = r .* cos.(θ) .* direction + rand(num_points_per_class) .* noise
        y = r .* sin.(θ) .* direction + rand(num_points_per_class) .* noise
        return hcat(x, y)
    end

    # Generate the two spirals
    spiral1 = generate_spiral(num_points_per_class, 1, noise)  # Clockwise
    spiral2 = generate_spiral(num_points_per_class, -1, noise)  # Counterclockwise

    # Combine the spirals and create labels
    points = vcat(spiral1, spiral2)
    labels = vcat(fill(1, num_points_per_class), fill(2, num_points_per_class))

    return points, labels
end

"""
    make_blobs_2d(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)

Generates a dataset of points distributed into distinct 2D blobs.
Each blob corresponds to a class.

# Arguments
- `num_classes`: Number of blobs (classes) to generate.
- `num_points_per_class`: Number of points per class.
- `noise`: Maximum noise applied to each point.

# Returns
- A tuple of `(points, labels)`, where `points` is a matrix of 2D points, and `labels` indicates class membership.

# Example
`points, labels = make_blobs_2d(5, 500, 0.5)`
"""
function make_blobs_2d(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)
    points = Matrix{Float64}(undef, 0, 2)
    labels = Vector{Int}(undef, 0)

    for class in 1:num_classes
        # Add noise and shift blob centers
        noise_values = noise .* (2 .* rand(num_points_per_class) .- 1)
        x = rand(num_points_per_class) .+ noise_values .+ 2 * class
        y = rand(num_points_per_class) .+ noise_values .+ 2 * class

        # Combine x and y
        points_for_class = hcat(x, y)
        points = vcat(points, points_for_class)
        labels = vcat(labels, fill(class, num_points_per_class))
    end

    return points, labels
end