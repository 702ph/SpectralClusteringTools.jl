export make_circles
export make_spirals_2d
export make_blobs_2d
export make_moons

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
    num_classes > 0 || throw(ArgumentError("Number of classes must be greater than 0"))
    num_points_per_class > 0 || throw(ArgumentError("Number of points per class must be greater than 0"))
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
    num_points_per_class > 0 || throw(ArgumentError("Number of points per class must be greater than 0"))
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
    num_classes > 0 || throw(ArgumentError("Number of classes must be greater than 0"))
    num_points_per_class > 0 || throw(ArgumentError("Number of points per class must be greater than 0"))
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

"""
    make_moons(n_samples::Int=300, noise::Float64=0.05)

Generate two interleaved moon-shaped datasets with overlapping regions. This function is primarily used for creating synthetic data to test clustering algorithms.

# Arguments
- `n_samples` (`Int=300`): Number of base samples for each moon shape
- `noise` (`Float64=0.05`): Standard deviation of Gaussian noise added to the data points

# Returns
Returns a tuple `(X, labels)`:
- `X`: `2×N` matrix where `N = 2 * (n_samples + n_samples÷3)`, each column represents a 2D data point
- `labels`: Vector of length `N` containing class labels (`1` or `2`) for each point

# Implementation Details
1. Generates base data points for two moon shapes.
2. Adds Gaussian noise to each data point.
3. Generates additional mixing points (`n_samples÷3` points) for each class.
4. Mixing points have higher noise (`2x` the base noise).

# Example
```julia
X, labels = make_moons(500, 0.1)
scatter(X[1,:], X[2,:], group=labels, title="Mixed Moons Dataset")
```
"""
function make_moons(n_samples::Int=300, noise::Float64=0.05)
    n_samples > 0 || throw(ArgumentError("Number of samples points must be greater than 0"))
    
    t = range(0, π, length=n_samples)
    x1 = cos.(t)
    y1 = sin.(t)

    x2 = 0.5 .- cos.(t) 
    y2 = 0.3 .- sin.(t)  
    

    x1 .+= randn(length(x1)) .* noise
    y1 .+= randn(length(y1)) .* noise
    x2 .+= randn(length(x2)) .* noise
    y2 .+= randn(length(y2)) .* noise
    
   
    n_mix = n_samples ÷ 3  
    

    mix_indices1 = rand(1:n_samples, n_mix)
    mix_x1 = x1[mix_indices1] .+ randn(n_mix) .* noise * 2
    mix_y1 = y1[mix_indices1] .+ randn(n_mix) .* noise * 2
    
    mix_indices2 = rand(1:n_samples, n_mix)
    mix_x2 = x2[mix_indices2] .+ randn(n_mix) .* noise * 2
    mix_y2 = y2[mix_indices2] .+ randn(n_mix) .* noise * 2
    
    X1 = hcat(x1, y1)
    X2 = hcat(x2, y2)
    Mix1 = hcat(mix_x2, mix_y2)
    Mix2 = hcat(mix_x1, mix_y1)
    
    X = vcat(X1, Mix1, X2, Mix2)
    labels = fill(1, size(X, 1))
    
    return Matrix(X'), labels
end