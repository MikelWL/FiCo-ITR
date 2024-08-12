# Similarity Measures in FiCo-ITR

## Overview

The `compute_similarity` function in FiCo-ITR provides efficient implementations of four similarity measures commonly used in image-text retrieval tasks: cosine similarity, Euclidean similarity, Hamming similarity, and inner product. This document outlines the considerations, implementation details, and usage guidelines for these similarity measures.

## Implemented Measures

### 1. Cosine Similarity

- **Formula**: cos(θ) = (x · y) / (||x|| * ||y||)
- **Range**: [-1, 1]
- **Interpretation**: 1 indicates maximum similarity, -1 indicates maximum dissimilarity, 0 indicates orthogonality
- **Implementation Notes**: 
  - Vectors are normalized before dot product for numerical stability
  - Efficient for high-dimensional sparse data

### 2. Euclidean Similarity

- **Formula**: similarity = 1 / (1 + sqrt(Σ(x_i - y_i)^2))
- **Range**: (0, 1]
- **Interpretation**: 1 indicates identical vectors, values close to 0 indicate high dissimilarity
- **Implementation Notes**:
  - Computed using an optimised formula to avoid explicit pairwise distances
  - A small epsilon (1e-8) is added to prevent division by zero
  - Transformed from distance to similarity for consistency with other measures

### 3. Hamming Similarity

- **Formula**: similarity = -(number of differing bits) / (total number of bits)
- **Range**: [-1, 0]
- **Interpretation**: 0 indicates identical bit strings, -1 indicates completely different bit strings
- **Implementation Notes**:
  - Assumes binary input vectors (0s and 1s)
  - Efficient for comparing binary feature vectors or hash codes

### 4. Inner Product

- **Formula**: x · y
- **Range**: Unbounded
- **Interpretation**: Higher values indicate more similarity
- **Implementation Notes**:
  - Simple dot product, efficient for dense vectors

## Performance Considerations

1. **Memory Efficiency**: 
   - The implementations avoid creating large intermediate arrays where possible

3. **Vectorisation**:
   - All implementations leverage NumPy's vectorised operations for efficiency

5. **Extremely large datasets**:
    - This implementation, although optimised, may not be the fastest for very large datasets.
    - For very large datasets, consider using a more efficient implementation based on FAISS or other such similarity search-centric libraries. This implementation is intended to enable typical benchmarking scenarios.

## Usage Guidelines

1. **Choosing a Similarity Measure**:
   - Cosine similarity is often preferred for text and image embeddings
   - Euclidean similarity is useful when the magnitude of vectors is important
   - Hamming similarity is ideal for binary feature vectors or hash codes
   - Inner product can be used when vector magnitudes are pre-normalized or irrelevant

2. **Input Preprocessing**:
   - Ensure input vectors are of the same dimensionality
   - For Hamming similarity, input vectors should be binary (0s and 1s)
