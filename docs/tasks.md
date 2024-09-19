# Tasks in FiCo-ITR

## Overview

The `tasks` module in FiCo-ITR is designed to perform various retrieval tasks for evaluating image-text retrieval models. It provides functions for both category-level and instance-level retrieval, handling the complexities of alignment between images, captions, and labels.

## Key Functions

### 1. Category Retrieval

```python
def category_retrieval(sim1: np.ndarray, labels: np.ndarray, k: Optional[int] = None, sim2: Optional[np.ndarray] = None) -> Tuple[float, float]:
```

This function performs category-level retrieval for both image-to-text and text-to-image directions.

#### Input:
- `sim1`: Precomputed similarity matrix for image-to-text
- `labels`: Category labels for the items
- `k`: Number of top results to consider (optional)
- `sim2`: Precomputed similarity matrix for text-to-image (optional)

#### Output:
A tuple containing:
- `mAP_i2t`: Mean Average Precision for image-to-text retrieval
- `mAP_t2i`: Mean Average Precision for text-to-image retrieval

#### Alignment Handling:
The function calculates alignment ratios between embeddings and labels, and aligns the labels accordingly using the `_calculate_ratios` and `_align_labels` helper functions.

### 2. Instance Retrieval

```python
def instance_retrieval(similarity_matrix: np.ndarray, t2i_sim = None) -> Tuple[Dict[str, float], Dict[str, float]]:
```

This function performs instance-level retrieval for both image-to-text and text-to-image directions.

#### Input:
- `similarity_matrix`: Precomputed similarity matrix for image-to-text
- `t2i_sim`: Precomputed similarity matrix for text-to-image (optional)

#### Output:
A tuple of two dictionaries, one for image-to-text and one for text-to-image retrieval, each containing:
- `R@1`, `R@5`, `R@10`: Recall at 1, 5, and 10
- `MedianR`: Median rank
- `MeanR`: Mean rank

#### Alignment Handling:
The function assumes a specific alignment where each image has 5 corresponding captions. It handles different shapes of similarity matrices for flexibility.

## Helper Functions

### 1. _calculate_ratios

```python
def _calculate_ratios(n_images: int, n_captions: int, n_labels: int) -> Tuple[int, int]:
```

Calculates alignment ratios between embeddings and labels.

### 2. _align_labels

```python
def _align_labels(labels: np.ndarray, img_ratio: int) -> np.ndarray:
```

Aligns labels with embeddings based on the calculated ratio.

### 3. _compute_map

```python
def _compute_map(similarity_matrix: np.ndarray, labels: np.ndarray, k: Optional[int] = None) -> float:
```

Computes the Mean Average Precision (mAP) for a given similarity matrix and labels.

### 4. instance_i2t

```python
def instance_i2t(similarity_matrix: np.ndarray) -> Dict[str, float]:
```

Performs image-to-text retrieval and computes various metrics.

### 5. instance_t2i

```python
def instance_t2i(similarity_matrix: np.ndarray) -> Dict[str, float]:
```

Performs text-to-image retrieval and computes various metrics.

## Metrics

1. **Mean Average Precision (mAP)**: Measures the average precision across all queries, considering the rank of all relevant items.

2. **Recall at K (R@K)**: The percentage of queries where a relevant item is found within the top K results.

3. **Median Rank (MedianR)**: The median position of the first relevant item in the ranked list of results.

4. **Mean Rank (MeanR)**: The average position of the first relevant item in the ranked list of results.

## Important Considerations

1. **Alignment**: The functions handle alignment ratios between images, captions, and labels. Non-integer ratios are handled by rounding down and issuing a warning.

2. **Similarity Matrix**: The instance retrieval functions can handle different shapes of similarity matrices, providing flexibility for various input formats.

3. **Performance**: For large datasets, consider the memory requirements, especially for the similarity matrix computations.

## Usage Examples

### Category Retrieval

```python
from fico_itr.tasks import category_retrieval
from fico_itr.similarity import compute_similarity

# Assume img_embs, txt_embs, and labels are loaded
sim1 = compute_similarity(img_embs, txt_embs)
mAP_i2t, mAP_t2i = category_retrieval(sim1, labels)
print(f"mAP (Image-to-Text): {mAP_i2t:.4f}")
print(f"mAP (Text-to-Image): {mAP_t2i:.4f}")
```

### Instance Retrieval

```python
from fico_itr.tasks import instance_retrieval
from fico_itr.similarity import compute_similarity

# Assume img_embs and txt_embs are loaded
similarity_matrix = compute_similarity(img_embs, txt_embs)
i2t_results, t2i_results = instance_retrieval(similarity_matrix)

print(f"Image-to-Text R@1: {i2t_results['R@1']:.2f}")
print(f"Text-to-Image R@1: {t2i_results['R@1']:.2f}")
```

## Limitations and Future Work

1. The current implementation assumes specific alignment ratios but provides warnings for non-integer ratios.
2. Performance optimizations for very large datasets may be needed.