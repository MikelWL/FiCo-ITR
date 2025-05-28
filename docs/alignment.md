# Image-Text Alignment in FiCo-ITR v1.0.0

## Overview

FiCo-ITR v1.0.0 handles various data formats encountered in image-text retrieval research. This document explains the alignment mechanisms, edge cases, and implementation details.

## Core Concepts

### 1. Matrix Orientations

FiCo-ITR expects similarity matrices in **images × captions** format, but automatically detects and corrects incorrect orientations:

```python
# Auto-transpose detection
if n_images > n_captions:  # Wrong orientation detected
    similarity_matrix = similarity_matrix.T
```

### 2. Caption Distribution Types

#### Uniform Distribution
- All images have the same number of captions
- Example: Flickr30k with exactly 5 captions per image
- Handled with integer `captions_per_image` parameter

#### Non-uniform Distribution
- Images have varying numbers of captions
- Example: COCO with 25010 captions for 5000 images
- Requires explicit caption-to-image mapping

### 3. Matrix Types

#### Rectangular Matrices (Standard)
- Different dimensions for images and captions
- Example: 1000×5000 for Flickr30k
- Most straightforward to handle

#### Square Matrices (Special Cases)
- Equal dimensions, often from image duplication
- Example: vsrn/ucch with 5000×5000 (1000 images duplicated 5×)
- Requires special handling to preserve ground truth

## Alignment Mechanisms

### 1. Caption Mapping Creation

The `_create_caption_mapping()` function handles multiple input formats:

```python
def _create_caption_mapping(captions_per_image, n_images, n_captions):
    """
    Returns: (caption_to_image_array, is_uniform, captions_per_image_value)
    """
```

#### Input Formats Supported:

1. **'auto'**: Attempts to infer uniform distribution
   - Validates divisibility: `n_captions % n_images == 0`
   - Throws informative error for non-uniform cases

2. **int**: Fixed captions per image
   - Validates: `n_images * captions_per_image == n_captions`
   - Most common case (e.g., 5 for Flickr30k)

3. **list/array of caption counts**: Per-image caption counts
   - Example: `[5, 5, 4, 6, 5, ...]` for 5 images
   - Builds mapping: `caption_idx → image_idx`

4. **list/array of image indices**: Direct caption-to-image mapping
   - Example: `[0,0,0,0,0, 1,1,1,1,1, ...]`
   - Length must equal `n_captions`

5. **dict**: Explicit `{caption_id: image_id}` mapping
   - Most flexible format
   - Sparse representation supported

### 2. Label Alignment in Category Retrieval

Category retrieval requires aligning labels with potentially duplicated embeddings:

```python
# For uniform distributions
if n_queries < n_retrievals:  # i2t direction
    if n_queries == n_labels * cap_ratio:
        # Square matrix with duplicated images
        aligned_query_labels = _align_labels(labels, cap_ratio)
    else:
        # Standard case
        aligned_query_labels = labels
    aligned_retrieval_labels = _align_labels(labels, cap_ratio)
```

### 3. Square Matrix Handling

Square matrices require special context-aware validation:

```python
if n_images == n_captions and captions_per_image > 1:
    # Validate it's actually from duplication
    if n_images % captions_per_image != 0:
        raise ValueError("Square matrix not divisible by captions_per_image")
    
    # Calculate actual unique images
    n_unique_images = n_images // captions_per_image
```

## Implementation Details

### 1. Instance Retrieval Alignment

#### Image-to-Text (i2t)
For square matrices with duplicated images:
```python
# Use first duplicate of each unique image as query
for index in range(n_unique_images):
    query_idx = duplication_factor * index
    d = similarity_matrix[query_idx]  # First duplicate only
    # Find best rank among all captions for this image
```

#### Text-to-Image (t2i)
For square matrices, extract unique images:
```python
# Extract indices of unique images (0, 5, 10, ...)
unique_img_indices = np.arange(0, n_rows, duplication_factor)

# For each caption, find its corresponding unique image
for cap_idx in range(n_captions):
    target_unique_img = cap_idx // captions_per_image
    # Compute similarity only with unique images
    d = similarity_matrix[unique_img_indices, cap_idx]
```

### 2. Non-uniform Distribution Handling

For datasets like COCO with 25010 test captions:

```python
# Using caption-to-image mapping
if n_queries < n_retrievals:  # i2t
    aligned_query_labels = labels  # Images already unique
    # Map each caption to its image's label
    aligned_retrieval_labels = labels[caption_to_image]
else:  # t2i
    # Map each caption query to its image's label
    aligned_query_labels = labels[caption_to_image]
    aligned_retrieval_labels = labels  # Images already unique
```

### 3. Multi-label Support

The system handles both single-label and multi-label scenarios:

```python
if labels.ndim == 1:
    # Single-label classification
    relevant = retrieved_labels == current_query_labels
else:
    # Multi-label classification (binary matrix)
    relevant = np.any(np.logical_and(retrieved_labels, current_query_labels), axis=1)
```

## Edge Cases and Validation

### 1. Divisibility Validation
```python
if n_rows % captions_per_image != 0:
    raise ValueError(
        f"Square matrix size {n_rows}×{n_cols} not divisible by "
        f"captions_per_image={captions_per_image}. For duplicated image matrices "
        f"(e.g., vsrn/ucch), use the original ratio (5 for COCO/Flickr30k)."
    )
```

### 2. Ambiguous Square Matrices
```python
if is_square and not is_uniform:
    raise ValueError(
        "Square matrix with non-uniform caption distribution is ambiguous. "
        "For duplicated image matrices, use captions_per_image=5. "
        "For true non-uniform distributions, provide rectangular matrix."
    )
```

### 3. Index Bounds Checking
```python
if np.any(indices >= len(aligned_retrieval_labels)):
    raise IndexError(
        f"Index {np.max(indices)} out of bounds for retrieval labels "
        f"of length {len(aligned_retrieval_labels)}"
    )
```

## Performance Optimizations

1. **Vectorized Operations**: All alignment operations use NumPy vectorization
2. **Lazy Transpose**: Matrices are only transposed when necessary
3. **Memory Efficiency**: Caption mappings use int32 for memory savings
4. **Early Validation**: Fail fast with informative errors before expensive computations

## Debugging Alignment Issues

Common symptoms and solutions:

1. **"Expected X captions but got Y"**
   - Check if using square matrix from duplicated images
   - Try setting `captions_per_image=5` (or appropriate ratio)

2. **"Uneven caption distribution detected"**
   - Dataset has non-uniform distribution (e.g., COCO)
   - Provide explicit caption-to-image mapping

3. **"Square matrix with non-uniform distribution is ambiguous"**
   - Either use uniform `captions_per_image` for duplicated images
   - Or provide rectangular matrix for true non-uniform data