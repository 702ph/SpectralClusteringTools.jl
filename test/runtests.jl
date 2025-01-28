using SpectralClusteringTools
using Test
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)

@testset "DataSets.jl" begin
    @testset "make_spheres tests" begin
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

        @testset "Point normalization: Scaling enabled" begin
            num_classes = 3
            num_points_per_class = 20
            noise = 0.1
            adjust_scale = true
            points, _ = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)
    
            expected_total_points = 0
            for class in 1:num_classes
                radius = class
                scale = adjust_scale ? (4π * radius^2) / (4π) : 1
                expected_total_points += Int(round(num_points_per_class * scale))
            end
    
            @test size(points, 1) == expected_total_points
            @test all(points .>= 0.0)
            @test all(points .<= 1.0)
        end

        @testset "Error handling" begin
            @test_throws ArgumentError make_spheres(3, -100)
        end
        
        @testset "Single class" begin
            num_classes = 1
            num_points_per_class = 10
            noise = 0.0
            adjust_scale = false
            points, labels = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)

            @test size(points, 1) == num_points_per_class
            @test all(labels .== 1)
        end
    end

    @testset "make_lines tests" begin
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

        @testset "Error handling" begin
            @test_throws ArgumentError make_lines(3, -100)
        end

        @testset "Single class" begin
            num_classes = 1
            num_points_per_class = 10
            noise = 0.1
            points, labels = make_lines(num_classes, num_points_per_class, noise)

            @test size(points, 1) == num_points_per_class
            @test all(labels .== 1)
        end
    end

    @testset "make_spirals tests" begin
        @testset "Basic functionality" begin
            num_points_per_class = 10
            noise = 0.1
            points, labels = make_spirals(num_points_per_class, noise)

            @test size(points, 1) == 2 * num_points_per_class
            @test size(points, 2) == 3
            @test length(labels) == size(points, 1)
            @test all(labels .>= 1 .&& labels .<= 2)
        end

        @testset "Very few points" begin
            num_points_per_class = 1
            noise = 0.0
            points, labels = make_spirals(num_points_per_class, noise)

            @test size(points, 1) == 2
            @test all(labels .== [1, 2])
        end
    end

    @testset "make_blobs tests" begin
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

        @testset "Error handling" begin
            @test_throws ArgumentError make_blobs(3, -100)
        end

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


@testset "spectral_clustering" begin
    @testset "Basic functionality:" begin
        @testset "ε-neighborhood graph" begin
            X = [0.0 1.0 0.0 -1.0; 1.0 0.0 -1.0 0.0]
            params = SpectralClusteringParams(:ε, 1, 1.5, 1.0)
            k = 3
            assignments = spectral_clustering(X, k, params)

            @test length(assignments) == size(X, 2)
            @test all(assignments .>= 1 .&& assignments .<= k)
        end

        @testset "k-nearest neighbor graph" begin
            X = [2.0 1.0 5.0 -1.0; 1.0 6.0 1.0 3.0]
            params = SpectralClusteringParams(:knn, 2, 1.5, 1.2)
            k = 3
            assignments = spectral_clustering(X, k, params)

            @test length(assignments) == size(X, 2)
            @test all(assignments .>= 1 .&& assignments .<= k)
        end

        @testset "Mutual k-nearest neighbor graph" begin
            num_classes = 3
            num_points_per_class = 10
            noise = 0.0
            adjust_scale = true

            X, _ = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)
            params = SpectralClusteringParams(:mutual_knn, 7, 7.0, 0.037)
            k = 3
            X_norm = (X .- mean(X, dims=1)) ./ std(X, dims=1)
            X_clustering = Matrix(X_norm')

            assignments = spectral_clustering(X_clustering, k, params)
    
            @test length(assignments) == size(X_clustering, 2)
            @test all(assignments .>= 1 .&& assignments .<= k)
        end

        @testset "Fully connected graph" begin
            X = [0.0 1.0 0.0 -1.0; 1.0 0.0 -1.0 0.0]
            params = SpectralClusteringParams(:fully_connected, 1, 0.0, 1.0)
            k = 2
            assignments = spectral_clustering(X, k, params)
    
            @test length(assignments) == size(X, 2)
            @test all(assignments .>= 1 .&& assignments .<= k)
        end
    end

    @testset "Error handling:" begin
        
        @testset "Invalid parameters" begin
            X = [0.0 1.0 0.0 -1.0; 1.0 0.0 -1.0 0.0]

            @test_throws ArgumentError spectral_clustering(X, -1, SpectralClusteringParams(:ε, 0, 1.5, 1.0))
            @test_throws ArgumentError spectral_clustering(X, 2, SpectralClusteringParams(:ε, 1, -1.5, 1.0))
            @test_throws ArgumentError spectral_clustering(X, 2, SpectralClusteringParams(:knn, -1, 1.0, 1.0))
        end

        @testset "Empty input" begin
            X = zeros(2, 0)
            params = SpectralClusteringParams(:fully_connected, 0, 0.0, 1.0)
            k = 2

            @test_throws ArgumentError spectral_clustering(X, k, params)
        end
    end

    @testset "Reproducibility test" begin
        Random.seed!(42)
        X = randn(2, 50)
        params = SpectralClusteringParams(:knn, 5, 1.5, 1.0)
        k = 3
        assignments1 = spectral_clustering(X, k, params)

        
        Random.seed!(42)
        X = randn(2, 50)
        params = SpectralClusteringParams(:knn, 5, 1.5, 1.0)
        k = 3

        assignments2 = spectral_clustering(X, k, params)
    
        @test assignments1 == assignments2
    end

    @testset "Noise robustness:" begin
        num_classes = 3
        num_points_per_class = 20
        noise_levels = [0.1, 0.5, 1.0]
        k = num_classes
    
        for noise in noise_levels
            X, _ = make_spheres(num_classes, num_points_per_class, noise, true)
            params = SpectralClusteringParams(:fully_connected, 1, 0.0, 1.0)
            X_norm = (X .- mean(X, dims=1)) ./ std(X, dims=1)
            X_clustering = Matrix(X_norm')
    
            assignments = spectral_clustering(X_clustering, k, params)
    
            @test length(assignments) == size(X_clustering, 2)
            @test all(assignments .>= 1 .&& assignments .<= k)
        end
    end

    @testset "Edge case:" begin
        @testset "Large dataset" begin
            X = randn(2, 1000)
            params = SpectralClusteringParams(:knn, 10, 1.0, 1.0)
            k = 3
            assignments = spectral_clustering(X, k, params)
    
            @test length(assignments) == size(X, 2)
            @test all(assignments .>= 1 .&& assignments .<= k)
        end

        @testset "High-dimensional data" begin
            X = randn(100, 50)
            params = SpectralClusteringParams(:fully_connected, 1, 0.0, 1.0)
            k = 3
            assignments = spectral_clustering(X, k, params)
        
            @test length(assignments) == size(X, 2)
            @test all(assignments .>= 1 .&& assignments .<= k)
        end
    end
end