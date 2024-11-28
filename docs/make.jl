using SpectralClusteringTools
using Documenter

DocMeta.setdocmeta!(SpectralClusteringTools, :DocTestSetup, :(using SpectralClusteringTools); recursive=true)

makedocs(;
    modules=[SpectralClusteringTools],
    authors="Sh F <fuchigami@campus.tu-berlin.de>",
    sitename="SpectralClusteringTools.jl",
    format=Documenter.HTML(;
        canonical="https://702ph.github.io/SpectralClusteringTools.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/702ph/SpectralClusteringTools.jl",
    devbranch="main",
)
