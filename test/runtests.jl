using SpectralClusteringTools
using Test
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)

@testset "spectral_clustering_NJW.jl" begin
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

@testset "Utils" begin
    @testset "DataSets.jl" begin
        @testset "make_spheres tests" begin
            @testset "Basic functionality: Scaling disabled" begin
                num_classes = 2
                num_points_per_class = 10
                noise = 0.1
                adjust_scale = false
                points, labels = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)
    
                @test size(points, 1) == num_classes * num_points_per_class
                @test size(points, 2) == 3
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
                @test all(isfinite.(points))
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
                @test all(isfinite.(points))
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
                @test all(isfinite.(points))
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
    
    @testset "2D_data_testing.jl" begin
        @testset "generate_mixed_concentric_data" begin
            @testset "Basic functionality" begin
                X, labels = generate_mixed_concentric_data()
            
                @test size(X, 1) == 2
                @test size(X, 2) == length(labels)
                @test all(labels .== 1)
                @test all(isfinite.(X))
            end
        end
    
        @testset "generate_mixed_moons_data" begin
            @testset "Basic functionality" begin
                X, labels = generate_mixed_moons_data(400, 0.1)
            
                @test size(X, 1) == 2
                @test size(X, 2) == length(labels)
                @test all(labels .== 1)
                @test all(isfinite.(X))
            end
    
            @testset "Error handling:" begin
                @testset "Invalid parameters" begin
                    @test_throws ArgumentError generate_mixed_moons_data(-400, 0.1)
                end
            end
    
            @testset "Edge case:" begin
                @testset "Small dataset" begin
                    # Extreme sample sizes
                    X, labels = generate_mixed_moons_data(2, 0.05)  # Very small dataset
                    @test size(X, 2) == length(labels)
                end
        
                @testset "Large dataset" begin
                    # Extreme sample sizes
                    X, labels = generate_mixed_moons_data(5000, 0.5)  # Very small dataset
                    @test size(X, 2) == length(labels)
                end
            end
        end
    end
    
    @testset "DataSets_2D.jl" begin
        @testset "make_circles tests" begin
            @testset "Basic functionality" begin
                num_classes = 3
                num_points_per_class = 1000
    
                points, labels = make_circles(num_classes, num_points_per_class, 0.1, false)
            
                @test size(points, 1) == num_classes * num_points_per_class
                @test length(labels) == size(points, 1)
                @test all(labels .<= num_classes)
                @test all(isfinite.(points))
            end
    
            @testset "Point normalization and scaling enabled" begin
                num_classes = 3
                num_points_per_class = 1000
                adjust_scale = true
                points, labels = make_circles(num_classes, num_points_per_class, 0.1, adjust_scale)
    
                @test all(points .>= 0.0)
                @test all(points .<= 1.0)
                @test size(points, 1) > num_classes * num_points_per_class
            end
            
            @testset "Single class" begin
                num_classes = 1
                num_points_per_class = 1000
    
                points, labels = make_circles(num_classes, num_points_per_class, 0.1, true)
    
                @test size(points, 1) == num_points_per_class
                @test all(labels .== 1)
            end
        end
    
        @testset "make_lines_2d tests" begin
            @testset "Basic functionality" begin
                num_classes = 3
                num_points_per_class = 1000
    
                points, labels = make_lines_2d(num_classes, num_points_per_class, 0.1)
            
                @test size(points, 1) == num_classes * num_points_per_class
                @test length(labels) == size(points, 1)
                @test all(labels .<= num_classes)
                @test all(isfinite.(points))
            end
            
            @testset "Single class" begin
                num_classes = 1
                num_points_per_class = 1000
    
                points, labels = make_lines_2d(num_classes, num_points_per_class, 0.1)
    
                @test size(points, 1) == num_points_per_class
                @test all(labels .== 1)
            end
        end
    
        @testset "make_spirals_2d tests" begin
            @testset "Basic functionality" begin
                num_points_per_class = 1000
    
                points, labels = make_spirals_2d(num_points_per_class, 0.1)
            
                @test size(points, 1) == 2 * num_points_per_class
                @test length(labels) == size(points, 1)
                @test all(labels .<= 2)
                @test all(isfinite.(points))
            end
        end
    
        @testset "make_blobs_2d tests" begin
            @testset "Basic functionality" begin
                num_classes = 3
                num_points_per_class = 1000
    
                points, labels = make_blobs_2d(num_classes, num_points_per_class, 0.1)
            
                @test size(points, 1) == num_classes * num_points_per_class
                @test length(labels) == size(points, 1)
                @test all(labels .<= num_classes)
                @test all(isfinite.(points))
            end
            
            @testset "Single class" begin
                num_classes = 1
                num_points_per_class = 1000
    
                points, labels = make_blobs_2d(num_classes, num_points_per_class, 0.1)
    
                @test size(points, 1) == num_points_per_class
                @test all(labels .== 1)
            end
        end
    end
    
    @testset "compare_results.jl" begin
        @testset "compute_accuracy tests" begin
            @test compute_accuracy([1, 2, 1], [1, 2, 1]) ≈ 1.0
            @test compute_accuracy([1, 2, 1], [1, 2, 3]) ≈ 2/3
            @test compute_accuracy([1, 1, 1], [2, 2, 2]) ≈ 0.0
            @test_throws ArgumentError compute_accuracy([1, 1, 0, 0], [0, 0, 1, 1, 1, 1])
        end
        
        @testset "contingency_matrix tests" begin
            matrix = contingency_matrix([1, 1, 0, 0], [0, 0, 1, 1])
            @test matrix == [2 0; 0 2] 
            @test_throws ArgumentError contingency_matrix([1, 1, 0, 0], [0, 0, 1, 1, 1, 1])
        end
        
        @testset "combinations tests" begin
            @test combinations(5, 3) == 10
            @test combinations(6, 2) == 15
            @test combinations(4, 4) == 1
            @test combinations(4, 5) == 0
            @test combinations(4, -5) == 0
        end
        
        @testset "compute_ari tests" begin
            @test compute_ari([1, 1, 0, 0], [0, 0, 1, 1]) ≈ 1.0
            @test_throws ArgumentError compute_ari([1, 1, 0, 0], [0, 0, 1, 1, 1, 1])
        end
        
        @testset "compute_nmi tests" begin
            @test compute_nmi([1, 1, 0, 0], [0, 0, 1, 1]) ≈ 1.0
            @test compute_nmi([1, 1, 2, 2], [1, 1, 2, 2]) ≈ 1.0
            @test_throws ArgumentError compute_nmi([1, 1, 0, 0], [0, 0, 1, 1, 1, 1])
        end
        
        @testset "print_efficiency tests" begin
            y_true = [1, 1, 0, 0]
            y_pred = [0, 0, 1, 1]
    
            expected_accuracy = compute_accuracy(y_true, y_pred)
            expected_ari = compute_ari(y_true, y_pred)
            expected_nmi = compute_nmi(y_true, y_pred)
        
            @test 0.0 <= expected_accuracy <= 1.0
            @test -1.0 <= expected_ari <= 1.0
            @test 0.0 <= expected_nmi <= 1.0
            
            @test print_efficiency(y_true, y_pred) == nothing
            @test_throws ArgumentError print_efficiency([1, 1, 0, 0], [0, 0, 1, 1, 1, 1])
        end
    end
end

@testset "self_tuning_spectral_clustering tests" begin
    @testset "Basic functionality" begin
        num_classes = 2
        num_points_per_class = 100
        noise = 0.1
        adjust_scale = false
        X, _ = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)
        X_norm = (X .- mean(X, dims=1)) ./ std(X, dims=1)
        X_clustering = Matrix(X_norm')
    
        max_C = 3
        params = SelfTuningParams(7, false)
    
        assignments, best_C, analysis_info = self_tuning_spectral_clustering(X_clustering, max_C, params)
    
        @test size(assignments, 1) == num_classes * num_points_per_class
        @test length(assignments) == size(X_clustering, 2)
        @test all(best_C .<= max_C)
        @test all(isfinite.(assignments))
        @test all(assignments .<= max_C)
    end

    @testset "is_spherical" begin
        num_classes = 3
        num_points_per_class = 10
        noise = 0.1
        adjust_scale = false
        X, _ = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)
        X_norm = (X .- mean(X, dims=1)) ./ std(X, dims=1)
        X_clustering = Matrix(X_norm')
    
        max_C = 3
        params = SelfTuningParams(7, true)
    
        assignments, best_C, analysis_info = self_tuning_spectral_clustering(X_clustering, max_C, params, is_spherical=true)
    
        @test size(assignments, 1) == num_classes * num_points_per_class
        @test length(assignments) == size(X_clustering, 2)
        @test all(best_C .<= max_C)
        @test all(isfinite.(assignments))
        @test all(assignments .<= max_C)
    end

    @testset "Error handling" begin
        @testset "Invalid parameters" begin
            num_classes = 2
            num_points_per_class = 10
            noise = 0.1
            adjust_scale = false
            X, _ = make_spheres(num_classes, num_points_per_class, noise, adjust_scale)
            X_norm = (X .- mean(X, dims=1)) ./ std(X, dims=1)
            X_clustering = Matrix(X_norm')

            @test_throws ArgumentError self_tuning_spectral_clustering(X_clustering, -1, SelfTuningParams(7, false))
            @test_throws ArgumentError self_tuning_spectral_clustering(X_clustering, 2, SelfTuningParams(-7, false))
        end
    end
end