using SpectralClusteringTools
using Test


@testset "make_sphere Tests" begin
  # Test: Basic functionality
  @testset "Basic functionality: Scaling disabled" begin
      num_classes = 2
      num_points_per_class = 10
      adjust_scale = false
      points, labels = make_sphere(num_classes, num_points_per_class, adjust_scale)
        
      @test size(points, 1) == num_classes * num_points_per_class
      @test size(points, 2) == 3  # Points are in 3D space
      @test length(labels) == size(points, 1)
      @test all(labels .>= 1 .&& labels .<= num_classes)
  end

  # Test: Scaling enabled
  @testset "Scaling enabled" begin
    num_classes = 3
    num_points_per_class = 20
    adjust_scale = true
    points, _ = make_sphere(num_classes, num_points_per_class, adjust_scale)
    
    # Calculate the expected number of points when scaling is applied
    expected_total_points = 0
    for class in 1:num_classes
        radius = class
        scale = adjust_scale ? (4π * radius^2) / (4π) : 1
        expected_total_points += Int(round(num_points_per_class * scale))
    end
    
    @test size(points, 1) == expected_total_points
    @test size(points, 2) == 3  # Points are in 3D space
    @test all(points .>= 0 .&& points .<= 1)  # Normalization to [0, 1]
  end

  # Test: Error handling
  @testset "Error handling" begin
      @test_throws ArgumentError make_sphere(-1, 100)  # Negative number of classes
      @test_throws ArgumentError make_sphere(3, -100)  # Negative number of points
  end
end