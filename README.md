# SpectralClusteringTools

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://702ph.github.io/SpectralClusteringTools.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://702ph.github.io/SpectralClusteringTools.jl/dev/)
[![Build Status](https://github.com/702ph/SpectralClusteringTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/702ph/SpectralClusteringTools.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/702ph/SpectralClusteringTools.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/702ph/SpectralClusteringTools.jl)


# SpectralClusteringTools

*Tools for Spectral Clustering.*

A package for creating test data and providing functions to facilitate clustering analysis.


## Package Features
- Tools for generating synthetic test datasets for clustering experiments.
- Functions for preprocessing data to prepare for spectral clustering.
- Implementation of spectral clustering algorithms.
- Utilities for visualizing clustering results and data distributions.


## Installation
This guide will help you set up and install the SpectralClusteringTools.jl package, which is developed as part of a university project for Julia version 1.10. Follow the steps below to ensure a successful installation.

### Step 1: Cloning the Git Repository
1. Open a terminal (command line interface) on your system.
2. Run the following commands to clone the repository and navigate to its directory:
```
git clone "https://github.com/702ph/SpectralClusteringTools.jl"
cd SpectralClusteringTools.jl

```
- The first command downloads the repository from GitHub to your local machine.
- The second command changes your current directory to the repository folder, where the package is located.


3. After navigating to the project folder, launch Julia by typing:
```
julia
```
This starts the Julia REPL (Read-Eval-Print Loop), an interactive environment where you can execute Julia commands.


### Step 2: Preparation for using the package
1. Inside the Julia REPL, enter the package manager mode by typing:
```
using Pkg
```

2.Activate the local environment for the project by running:
```
Pkg.activate(".")
```
- The `activate()` command tells Julia to use the package environment defined in the current directory (`.` refers to the current folder where the project was cloned).

3. Install all dependencies specified in the project's Project.toml file:
```
Pkg.instantiate()
```
The `instantiate()` command ensures all required packages for the project are downloaded and installed.



### Notes and Troubleshooting
- *Julia Version*: This project is developed and tested with *Julia version 1.10*. Ensure that you have this version installed on your system. You can check your Julia version by running:
```
julia --version
```
- *Dependencies*: The Pkg.instantiate() command will automatically install all required dependencies. If you encounter issues, try running Pkg.update() to ensure you have the latest compatible versions of the packages.


## Example
Please refer to the example section in the documentation.
[docs-stable-url]


## Documentation
- [**STABLE**][docs-stable-url] &mdash; **documentation of the most recently tagged version.**
- [**DEVEL**][docs-dev-url] &mdash; *documentation of the in-development version.*
