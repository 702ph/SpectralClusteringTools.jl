using SpectralClusteringTools
using Test

@testset "SpectralClusteringTools.jl" begin
    @testset "timestwo" begin
        @test timestwo(4.0) == 8.0
        @test timestwo(2.0) != 8.0
        # @test timestwo(3.0) == 8.0
      end
end
