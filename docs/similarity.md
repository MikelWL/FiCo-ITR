# Similarity Measures in FiCo-ITR v1.0.0

## Overview

The `compute_similarity` function implements four similarity measures for image-text retrieval evaluation.

## Implemented Measures

### 1. Cosine Similarity
**Formula**: `cos(θ) = (x · y) / (||x|| * ||y||)`
- **Range**: [-1, 1]
- **Implementation**: Normalizes vectors before dot product
- **Use case**: Standard for normalized embeddings

### 2. Euclidean Similarity
**Formula**: `similarity = 1 / (1 + sqrt(Σ(x_i - y_i)²) + ε)`
- **Range**: (0, 1]
- **Implementation**: Uses expansion `||x-y||² = ||x||² + ||y||² - 2x·y` to avoid explicit difference computation
- **Numerical stability**: `np.maximum(distances, 0)` handles floating point errors; epsilon (1e-8) prevents division by zero

### 3. Hamming Similarity
**Formula**: `similarity = -(number of differing elements) / (vector length)`
- **Range**: [-1, 0]
- **Implementation**: Element-wise comparison, normalized by vector length
- **Use case**: Binary or discrete feature vectors

### 4. Inner Product
**Formula**: `similarity = Σ(x_i * y_i)`
- **Range**: Unbounded
- **Implementation**: Direct matrix multiplication
- **Use case**: When vector magnitudes are meaningful

## Implementation Details

The function signature:
```python
def compute_similarity(
    x: np.ndarray,
    y: np.ndarray,
    measure: Literal['cosine', 'euclidean', 'hamming', 'inner_product'] = 'cosine'
) -> np.ndarray
```

**Input**: 
- `x`: Shape (n, d) - First set of vectors
- `y`: Shape (m, d) - Second set of vectors
- `measure`: Similarity measure to use

**Output**: Similarity matrix of shape (n, m)

## Computational Complexity

All measures have O(n×m×d) time complexity where:
- n = number of vectors in x
- m = number of vectors in y  
- d = dimensionality

Cosine and Euclidean have additional O(n×d) and O(m×d) operations for normalization/magnitude computation.

## Notes

- For very large (>1M comparisons), specialized libraries like FAISS may be more appropriate
- The implementation prioritizes correctness of task implementation and ease of use over maximum performance
- All measures return higher values for more similar vectors (Hamming uses negative values to maintain this convention)