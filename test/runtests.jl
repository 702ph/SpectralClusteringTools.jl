using SpectralClusteringTools
using Test
using Random

Random.seed!(42)

@testset "SpectralClusteringTools.jl" begin
    @testset "make_spheres tests" begin
        # Test: Basic functionality
        @testset "Basic functionality: Scaling disabled" begin
            num_classes = 2
            num_points_per_class = 10
            noise = 0.1
            adjust_scale = false
            points, labels = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)

            @test size(points, 1) == num_classes * num_points_per_class
            @test size(points, 2) == 3  # Points are in 3D space
            @test length(labels) == size(points, 1)
            @test all(labels .>= 1 .&& labels .<= num_classes)
        end

        # Test: Scaling enabled
        @testset "Point normalization: Scaling enabled" begin
            num_classes = 3
            num_points_per_class = 20
            noise = 0.1
            adjust_scale = true
            points, _ = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)
    
            # Calculate the expected number of points when scaling is applied
            expected_total_points = 0
            for class in 1:num_classes
                radius = class
                scale = adjust_scale ? (4π * radius^2) / (4π) : 1
                expected_total_points += Int(round(num_points_per_class * scale))
            end
    
            @test size(points, 1) == expected_total_points
            @test all(points .>= 0.0)  # All values should be >= 0
            @test all(points .<= 1.0)  # All values should be <= 1
        end

        # Test: Error handling
        @testset "Error handling" begin
            @test_throws ArgumentError make_spheres(3, -100)  # Negative number of points
        end
         # Edge case: Single class
        @testset "Single class" begin
            num_classes = 1
            num_points_per_class = 10
            noise = 0.0
            adjust_scale = false
            points, labels = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)

            @test size(points, 1) == num_points_per_class
            @test all(labels .== 1)  # All points should belong to the single class
        end
    end

    # Tests for make_lines
    @testset "make_lines tests" begin
        # Test: Basic functionality
        @testset "Basic functionality" begin
            num_classes = 2
            num_points_per_class = 10
            noise = 0.1
            points, labels = make_lines(num_classes, num_points_per_class, noise)

            @test size(points, 1) == num_classes * num_points_per_class
            @test size(points, 2) == 3
            @test length(labels) == size(points, 1)
            @test all(labels .>= 1 .&& labels .<= num_classes)
        end

        # Test: Error handling
        @testset "Error handling" begin
            @test_throws ArgumentError make_lines(3, -100)
        end

        # Edge case: Single class
        @testset "Single class" begin
            num_classes = 1
            num_points_per_class = 10
            noise = 0.1
            points, labels = make_lines(num_classes, num_points_per_class, noise)

            @test size(points, 1) == num_points_per_class
            @test all(labels .== 1)
        end
    end

    # Tests for make_spirals
    @testset "make_spirals tests" begin
        # Test: Basic functionality
        @testset "Basic functionality" begin
            num_points_per_class = 10
            noise = 0.1
            points, labels = make_spirals(num_points_per_class, noise)

            @test size(points, 1) == 2 * num_points_per_class
            @test size(points, 2) == 3
            @test length(labels) == size(points, 1)
            @test all(labels .>= 1 .&& labels .<= 2)
        end

        # Edge case: Very few points
        @testset "Very few points" begin
            num_points_per_class = 1
            noise = 0.0
            points, labels = make_spirals(num_points_per_class, noise)

            @test size(points, 1) == 2
            @test all(labels .== [1, 2])
        end
    end

    # Tests for make_blobs
    @testset "make_blobs tests" begin
        # Test: Basic functionality
        @testset "Basic functionality" begin
            num_classes = 2
            num_points_per_class = 10
            noise = 0.1
            points, labels = make_blobs(num_classes, num_points_per_class, noise)

            @test size(points, 1) == num_classes * num_points_per_class
            @test size(points, 2) == 3
            @test length(labels) == size(points, 1)
            @test all(labels .>= 1 .&& labels .<= num_classes)
        end

        # Test: Error handling
        @testset "Error handling" begin
            @test_throws ArgumentError make_blobs(3, -100)
        end

        # Edge case: Single class
        @testset "Single class" begin
            num_classes = 1
            num_points_per_class = 10
            noise = 0.1
            points, labels = make_blobs(num_classes, num_points_per_class, noise)

            @test size(points, 1) == num_points_per_class
            @test all(labels .== 1)
        end
    end
end