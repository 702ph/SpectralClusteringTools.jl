var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SpectralClusteringTools","category":"page"},{"location":"#SpectralClusteringTools","page":"Home","title":"SpectralClusteringTools","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tools for Spectral Clustering.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A package for creating test data and providing functions to facilitate clustering analysis.","category":"page"},{"location":"#Documentation","page":"Home","title":"Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SpectralClusteringTools's interface.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SpectralClusteringTools]","category":"page"},{"location":"#SpectralClusteringTools.SpectralClusteringParams","page":"Home","title":"SpectralClusteringTools.SpectralClusteringParams","text":"Construct a similarity graph. W: weighted adjacency matrix(A Tutorial on Spectral Clustering P7)\n\n● According to the reference, there are three ways to do it:     (1) The ε-neighborhood graph     (2) k-nearest neighbor graphs     (3) The fully connected graph ● Parameters for Spectral clustering\n\n\n\n\n\n","category":"type"},{"location":"#SpectralClusteringTools.fully_connected_graph-Tuple{Matrix{Float64}, Float64}","page":"Home","title":"SpectralClusteringTools.fully_connected_graph","text":"● Construct fully connected graph ● Connect all points with positive similarity with each other ● Weight all edge by sij ● Only useful if the similarity function itself models local neighbors， example: Gaussian\n\n\n\n\n\n","category":"method"},{"location":"#SpectralClusteringTools.make_blobs","page":"Home","title":"SpectralClusteringTools.make_blobs","text":"make_blobs(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)\n\nGenerates a dataset of points distributed on different blobs.  Each blob corresponds to a class.\n\nArguments\n\nnum_classes: Number of spheres (classes) to generate.\nnum_points_per_class: Number of points per class.\nnoise: Maximal noise applied for each point.\n\nReturns\n\nA tuple of (normalized_points, labels), where normalized_points is a matrix of 3D points, and labels indicates class membership.\n\nExample\n\npoints, labels = make_blobs(5, 500, 0.5)\n\n\n\n\n\n","category":"function"},{"location":"#SpectralClusteringTools.make_lines","page":"Home","title":"SpectralClusteringTools.make_lines","text":"make_lines(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0)\n\nGenerates a dataset of points distributed following straight lines. Each line corresponds to a class.\n\nArguments\n\nnum_classes: Number of lines (classes) to generate.\nnum_points_per_class: Number of points per class.\nnoise: Maximal noise applied for each point.\n\nReturns\n\nA tuple of (normalized_points, labels), where normalized_points is a matrix of 3D points, and labels indicates class membership.\n\nExample\n\npoints, labels = make_lines(5, 500, 0.3)\n\n\n\n\n\n","category":"function"},{"location":"#SpectralClusteringTools.make_spheres","page":"Home","title":"SpectralClusteringTools.make_spheres","text":"make_spheres(num_classes::Int, num_points_per_class::Int, noise::Float64=0.0, adjust_scale::Bool=true)\n\nGenerates a dataset of points distributed on the surfaces of spheres.  Each sphere corresponds to a class, and points are normalized.\n\nArguments\n\nnum_classes: Number of spheres (classes) to generate.\nnum_points_per_class: Number of points per class.\nnoise: Maximal noise applied for each point.\nadjust_scale: Adjusts the number of points.\n\nReturns\n\nA tuple of (normalized_points, labels), where normalized_points is a matrix of 3D points, and labels indicates class membership.\n\nExample\n\npoints, labels = make_sphere(3, 1000, 0.1, false)\n\n\n\n\n\n","category":"function"},{"location":"#SpectralClusteringTools.make_spirals","page":"Home","title":"SpectralClusteringTools.make_spirals","text":"make_spirals(num_points_per_class::Int, noise::Float64=0.0)\n\nGenerates a dataset of points following 2 interlacing spirals. Each spiral corresponds to a class.\n\nArguments\n\nnum_points_per_class: Number of points per class.\nnoise: Maximal noise applied for each point.\n\nReturns\n\nA tuple of (normalized_points, labels), where normalized_points is a matrix of 3D points, and labels indicates class membership.\n\nExample\n\npoints, labels = make_spirals(500, 1.0)\n\n\n\n\n\n","category":"function"}]
}
