export compute_accuracy
export compute_ari
export compute_nmi
export print_efficiency

# for testing the functions
export contingency_matrix
export combinations


"""
    compute_accuracy(y_true::Vector{Int}, y_pred::Vector{Int})

Computes the accuracy of predicted labels compared to true labels. 
The accuracy is defined as the proportion of correctly predicted labels to the total number of labels.

# Arguments
- `y_true`: A vector of true cluster labels.
- `y_pred`: A vector of predicted cluster labels.
  Both vectors must have the same length.

# Returns
- The accuracy as a Float64 value, representing the fraction of correctly classified instances.

# Example
`accuracy = compute_accuracy([1, 2, 1], [1, 2, 3])`
"""
function compute_accuracy(y_true::Vector{Int}, y_pred::Vector{Int})
	if length(y_true) != length(y_pred)
        throw(ArgumentError("Labels and predicted labels should have the same size"))
    end
	
	true_count = 0
	
	for i in 1:length(y_true)
        if y_true[i] == y_pred[i]
            true_count += 1
        end
    end
	
	accuracy = true_count / length(y_true)
	return accuracy
end


"""
    contingency_matrix(y_true::Vector{Int}, y_pred::Vector{Int})

Builds a contingency matrix to measure the overlap between two clusterings. 
Each entry (i, j) in the matrix represents the number of samples in both true cluster i and predicted cluster j.

# Arguments
- `y_true`: A vector of true cluster labels.
- `y_pred`: A vector of predicted cluster labels. Both vectors must have the same length.

# Returns
- A 2D matrix where rows correspond to true clusters and columns to predicted clusters, containing counts of overlapping elements.

# Example
`matrix = contingency_matrix([1, 1, 0, 0], [0, 0, 1, 1])
"""
function contingency_matrix(y_true::Vector{Int}, y_pred::Vector{Int})
    if length(y_true) != length(y_pred)
        throw(ArgumentError("Labels and predicted labels should have the same size"))
    end

    classes_true = unique(y_true)
    classes_pred = unique(y_pred)
    matrix = zeros(Int, length(classes_true), length(classes_pred))

    for (i, c1) in enumerate(classes_true)
        for (j, c2) in enumerate(classes_pred)
            matrix[i, j] = sum((y_true .== c1) .& (y_pred .== c2))
        end
    end

    return matrix
end


"""
    combinations(n::Int, k::Int)

Computes the number of ways to choose `k` elements from a set of `n` elements without considering the order.

# Arguments
- `n`: Total number of elements in the set (Integer).
- `k`: Number of elements to choose (Integer).

# Returns
- An integer representing the number of combinations. If `k > n` or `k < 0`, returns 0.

# Example
`c = combinations(5, 3)
"""
function combinations(n::Int, k::Int)
    if n < k || k < 0
        return 0
    elseif k == 0 || n == k
        return 1
    else
        num = prod(n-k+1:n)
        denom = prod(1:k)
        return num / denom
    end
end


"""
    compute_ari(y_true::Vector{Int}, y_pred::Vector{Int})

Computes the Adjusted Rand Index (ARI) to measure the similarity between two clusterings, accounting for chance.
The ARI ranges from -1 to 1, where 1 indicates perfect agreement, 0 indicates random labeling, and negative values suggest disagreement.

# Arguments
- `y_true`: A vector of true cluster labels.
- `y_pred`: A vector of predicted cluster labels. 
  Both vectors must have the same length.

# Returns
- The Adjusted Rand Index as a Float64 value.

# Example
`ari = compute_ari([1, 1, 0, 0], [0, 0, 1, 1])`
"""
function compute_ari(y_true::Vector{Int}, y_pred::Vector{Int})
    if length(y_true) != length(y_pred)
        throw(ArgumentError("Labels and predicted labels should have the same size"))
    end

    matrix = contingency_matrix(y_true, y_pred)
    n = sum(matrix)
    sum_comb = sum(sum(combinations(x, 2) for x in matrix))
    sum_row_comb = sum(combinations(x, 2) for x in vec(sum(matrix, dims=2)))
    sum_col_comb = sum(combinations(x, 2) for x in vec(sum(matrix, dims=1)))

    expected_index = (sum_row_comb * sum_col_comb) / combinations(n, 2)
    max_index = (sum_row_comb + sum_col_comb) / 2
    actual_index = sum_comb

    return (actual_index - expected_index) / (max_index - expected_index)
end


"""
    compute_nmi(y_true::Vector{Int}, y_pred::Vector{Int})

Computes the Normalized Mutual Information (NMI) between two clusterings to evaluate their similarity. 
The NMI ranges from 0 to 1, where 1 indicates perfect agreement and 0 indicates no mutual information.

# Arguments
- `y_true`: A vector of true cluster labels.
- `y_pred`: A vector of predicted cluster labels.
  Both vectors must have the same length.

# Returns
- The Normalized Mutual Information as a Float64 value.

# Example
`nmi = compute_nmi([1, 1, 0, 0], [0, 0, 1, 1])
"""
function compute_nmi(y_true::Vector{Int}, y_pred::Vector{Int})
    if length(y_true) != length(y_pred)
        throw(ArgumentError("Labels and predicted labels should have the same size"))
    end

    matrix = contingency_matrix(y_true, y_pred)
    n = sum(matrix)
    row_sums = vec(sum(matrix, dims=2))
    col_sums = vec(sum(matrix, dims=1))

    mutual_info = 0.0
    for i in 1:size(matrix, 1)
        for j in 1:size(matrix, 2)
            if matrix[i, j] > 0
                mutual_info += matrix[i, j] / n * log((matrix[i, j] * n) / (row_sums[i] * col_sums[j]))
            end
        end
    end

    h_true = -sum((row_sums ./ n) .* log.(row_sums ./ n))
    h_pred = -sum((col_sums ./ n) .* log.(col_sums ./ n))

    return mutual_info / sqrt(h_true * h_pred)
end


"""
    print_efficiency(y_true::Vector{Int}, y_pred::Vector{Int})

Computes the efficiency metrics for clustering, including Accuracy, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI). 

# Arguments
- `y_true`: A vector of true cluster labels.
- `y_pred`: A vector of predicted cluster labels.
  Both vectors must have the same length.

# Returns
- A `nothing` value (prints the efficiency metrics directly).

# Example
`print_efficiency([1, 1, 0, 0], [0, 0, 1, 1])
"""
function print_efficiency(y_true::Vector{Int}, y_pred::Vector{Int})
    if length(y_true) != length(y_pred)
        throw(ArgumentError("Labels and predicted labels should have the same size"))
    end

    accuracy = compute_accuracy(y_true, y_pred)
    ari = compute_ari(y_true, y_pred)
    nmi = compute_nmi(y_true, y_pred)

    println("Model Efficiency Metrics:")
    println("---------------------------")
    println("Accuracy: $(round(accuracy, digits=4))")
    println("ARI: $(round(ari, digits=4))")
    println("NMI: $(round(nmi, digits=4))")
end