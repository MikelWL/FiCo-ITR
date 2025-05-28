# Tasks in FiCo-ITR v1.0.0

## Overview

The `tasks` module implements retrieval evaluation algorithms for image-text retrieval benchmarks. This document provides technical details about the evaluation metrics, implementation strategies, and algorithmic optimizations.

## Core Functions

### 1. Instance Retrieval

```python
def instance_retrieval(
    similarity_matrix: np.ndarray, 
    t2i_sim: Optional[np.ndarray] = None,
    captions_per_image: Union[str, int, List, np.ndarray, Dict] = 'auto'
) -> Tuple[Dict[str, float], Dict[str, float]]
```

**Purpose**: Evaluates retrieval performance at the instance level, where each image-caption pair is considered individually.

**Parameters**:
- `similarity_matrix`: Primary similarity matrix (images × captions expected)
- `t2i_sim`: Optional separate matrix for text-to-image retrieval
- `captions_per_image`: Caption distribution specification (see Alignment section)

**Returns**: Tuple of dictionaries containing R@1, R@5, R@10, MedianR, and MeanR for both i2t and t2i directions.

### 2. Category Retrieval

```python
def category_retrieval(
    sim1: np.ndarray,
    labels: np.ndarray,
    k: Optional[int] = None,
    sim2: Optional[np.ndarray] = None,
    captions_per_image: Union[str, int, List, np.ndarray, Dict] = 'auto'
) -> Tuple[float, float]
```

**Purpose**: Evaluates retrieval performance at the category level using mean Average Precision (mAP).

**Parameters**:
- `sim1`: Primary similarity matrix
- `labels`: Category labels (single-label or multi-label)
- `k`: Top-k cutoff for mAP calculation
- `sim2`: Optional separate matrix for opposite direction
- `captions_per_image`: Caption distribution specification

**Returns**: Tuple of mAP scores for i2t and t2i directions.

## Evaluation Metrics

### 1. Recall at K (R@K)

**Definition**: Percentage of queries where at least one relevant item appears in top K results.

**Implementation**:
```python
def recall_at_k(ranks, k):
    return 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
```

**Properties**:
- **Range**: [0, 100]
- **Interpretation**: Higher is better
- **Common K values**: 1, 5, 10
- **Sensitivity**: Most sensitive to ranking quality at top positions

### 2. Median Rank (MedianR)

**Definition**: Median position of first relevant item across all queries.

**Implementation**:
```python
median_rank = np.floor(np.median(ranks)) + 1  # 1-indexed
```

**Properties**:
- **Range**: [1, n_items]
- **Interpretation**: Lower is better
- **Robustness**: Less sensitive to outliers than mean rank
- **Use case**: Good for detecting consistent performance

### 3. Mean Rank (MeanR)

**Definition**: Average position of first relevant item across all queries.

**Implementation**:
```python
mean_rank = ranks.mean() + 1  # 1-indexed
```

**Properties**:
- **Range**: [1, n_items]
- **Interpretation**: Lower is better
- **Sensitivity**: Affected by outliers
- **Use case**: Captures overall ranking quality

### 4. Mean Average Precision (mAP)

**Definition**: Average of precision values at each relevant item position.

**Detailed Implementation**:
```python
def compute_map(similarity_matrix, labels, k=None):
    ap_scores = []
    
    for query_idx in range(n_queries):
        # Get sorted retrieval indices
        sorted_indices = np.argsort(-similarity_matrix[query_idx])[:k]
        
        # Get relevance mask
        query_label = labels[query_idx]
        retrieved_labels = labels[sorted_indices]
        
        if labels.ndim == 1:  # Single-label
            relevant = retrieved_labels == query_label
        else:  # Multi-label
            relevant = np.any(
                np.logical_and(retrieved_labels, query_label), 
                axis=1
            )
        
        # Calculate average precision
        relevant_positions = np.where(relevant)[0]
        if len(relevant_positions) > 0:
            # Precision at each relevant position
            precisions = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
            ap = np.mean(precisions[relevant])
        else:
            ap = 0.0
        
        ap_scores.append(ap)
    
    return np.mean(ap_scores)
```

**Properties**:
- **Range**: [0, 1]
- **Interpretation**: Higher is better (1 = perfect ranking)
- **Comprehensiveness**: Considers all relevant items, not just first
- **Balance**: Rewards both precision and recall

## Implementation Details

### 1. Ranking Computation for Instance Retrieval

#### Image-to-Text (Standard Case)
```python
def compute_i2t_ranks(similarity_matrix, caption_mapping):
    ranks = np.zeros(n_images)
    
    for img_idx in range(n_images):
        # Get similarities for this image
        similarities = similarity_matrix[img_idx]
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Find target captions for this image
        if is_uniform:
            target_captions = range(
                img_idx * captions_per_image, 
                (img_idx + 1) * captions_per_image
            )
        else:
            # Non-uniform: find captions mapped to this image
            target_captions = np.where(caption_mapping == img_idx)[0]
        
        # Find best rank among target captions
        best_rank = 1e20  # Large initial value
        for cap_idx in target_captions:
            position = np.where(sorted_indices == cap_idx)[0][0]
            best_rank = min(best_rank, position)
        
        ranks[img_idx] = best_rank
    
    return ranks
```

#### Text-to-Image (Standard Case)
```python
def compute_t2i_ranks(similarity_matrix, caption_mapping):
    # Transpose for caption queries
    sim_t2i = similarity_matrix.T
    ranks = np.zeros(n_captions)
    
    for cap_idx in range(n_captions):
        similarities = sim_t2i[cap_idx]
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Find correct image for this caption
        if is_uniform:
            correct_img = cap_idx // captions_per_image
        else:
            correct_img = caption_mapping[cap_idx]
        
        # Find rank of correct image
        rank = np.where(sorted_indices == correct_img)[0][0]
        ranks[cap_idx] = rank
    
    return ranks
```

### 2. Special Handling for Square Matrices

Square matrices from duplicated images require unique approaches:

#### Image-to-Text with Duplicated Images
```python
# vsrn/ucch approach: Use first duplicate only
for unique_img_idx in range(n_unique_images):
    # Query with first duplicate (indices 0, 5, 10, ...)
    query_idx = unique_img_idx * duplication_factor
    similarities = similarity_matrix[query_idx]
    
    # Find best rank among all captions for this unique image
    # (captions are still indexed by original image duplicates)
```

#### Text-to-Image with Duplicated Images
```python
# Extract unique images for ranking
unique_img_indices = np.arange(0, n_total_images, duplication_factor)

for cap_idx in range(n_captions):
    # Compare only against unique images
    similarities = similarity_matrix[unique_img_indices, cap_idx]
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Find which unique image this caption belongs to
    target_unique_img = cap_idx // captions_per_image
    rank = np.where(sorted_indices == target_unique_img)[0][0]
```

### 3. Direction Detection and Validation

The system automatically determines retrieval direction:

```python
def determine_direction(n_queries, n_retrievals, n_labels, cap_ratio):
    if n_queries < n_retrievals:
        # Likely i2t: fewer images than captions
        direction = 'i2t'
        expected_queries = [n_labels, n_labels * cap_ratio]
        expected_retrievals = n_labels * cap_ratio
    else:
        # Likely t2i: more captions than images
        direction = 't2i'
        if n_queries == n_retrievals:  # Square matrix
            expected_queries = n_labels * cap_ratio
            expected_retrievals = n_labels * cap_ratio
        else:
            expected_queries = n_labels * cap_ratio
            expected_retrievals = n_labels
    
    return direction, expected_queries, expected_retrievals
```

## Optimization Strategies

### 1. Memory-Efficient Ranking

Instead of storing full sorted matrices:
```python
# Memory-intensive approach (avoid)
all_sorted = np.argsort(-similarity_matrix, axis=1)

# Memory-efficient approach (preferred)
for i in range(n_queries):
    sorted_indices = np.argsort(-similarity_matrix[i])
    # Process immediately, don't store
```

### 2. Early Stopping for Recall Metrics

For R@K metrics, we can stop once we find a match:
```python
def find_first_relevant_rank(sorted_indices, targets, max_k=10):
    for rank, idx in enumerate(sorted_indices[:max_k]):
        if idx in targets:
            return rank
    return max_k  # Not found in top K
```

### 3. Vectorized Label Alignment

For category retrieval with uniform distributions:
```python
# Slow: Loop-based alignment
aligned = []
for label in labels:
    aligned.extend([label] * ratio)

# Fast: Vectorized alignment
aligned = np.repeat(labels, ratio, axis=0)
```


## Edge Cases and Robustness

### 1. Empty Retrieval Results
```python
if len(relevant_indices) == 0:
    # No relevant items found
    ap = 0.0  # Defined behavior for mAP
    rank = n_items  # Worst possible rank
```

### 2. Duplicate Similarities
```python
# When multiple items have identical similarities
sorted_indices = np.argsort(-similarities, kind='stable')
# 'stable' ensures consistent ordering
```

### 3. Numerical Precision
```python
# For very large datasets, use float64 for ranks
ranks = np.zeros(n_queries, dtype=np.float64)

# For similarity comparisons
epsilon = 1e-8
if abs(sim1 - sim2) < epsilon:
    # Treat as equal
```

### 4. Label Validation
```python
# Ensure labels match data dimensions
if len(labels) not in [n_images, n_unique_images]:
    raise ValueError(
        f"Label count {len(labels)} doesn't match "
        f"images {n_images} or unique images {n_unique_images}"
    )
```


## Integration Examples

### 1. Standard Evaluation Pipeline
```python
# Load embeddings
img_emb = np.load('image_embeddings.npy')
txt_emb = np.load('text_embeddings.npy')
labels = np.load('category_labels.npy')

# Compute similarity
sim = compute_similarity(img_emb, txt_emb, measure='cosine')

# Instance-level evaluation
i2t_results, t2i_results = instance_retrieval(sim)
print(f"I2T R@1: {i2t_results['R@1']:.1f}%")
print(f"T2I R@1: {t2i_results['R@1']:.1f}%")

# Category-level evaluation
i2t_map, t2i_map = category_retrieval(sim, labels)
print(f"I2T mAP: {i2t_map:.3f}")
print(f"T2I mAP: {t2i_map:.3f}")
```

### 2. Handling Special Cases
```python
# For COCO with non-uniform distribution
caption_indices = np.load('coco_caption_to_image_mapping.npy')
i2t_results, t2i_results = instance_retrieval(
    sim, 
    captions_per_image=caption_indices
)

# For vsrn with square matrix
i2t_results, t2i_results = instance_retrieval(
    sim_square,  # 5000×5000
    captions_per_image=5  # Original ratio
)
```

### 3. Custom Evaluation Protocols
```python
# Top-100 mAP for large-scale retrieval
i2t_map100, t2i_map100 = category_retrieval(
    sim, 
    labels, 
    k=100
)

# Asymmetric evaluation
i2t_results = instance_i2t(sim_cosine)
t2i_results = instance_t2i(sim_euclidean)
```

