# Tasks in FiCo-ITR

## Overview

The `tasks` module in FiCo-ITR is designed to perform various retrieval tasks for evaluating image-text retrieval models. It provides functions for both category-level and instance-level retrieval, handling the complexities of alignment between images, captions, and labels.

## Key Functions

### 1. Category Retrieval

```python
def category_retrieval(similarity_matrix: np.ndarray, labels: np.ndarray, k: Optional[int] = None) -> dict:
```

This function performs category-level retrieval, which evaluates how well the model retrieves items from the same category.

#### Input:
- `similarity_matrix`: A precomputed similarity matrix between images and captions
- `labels`: Category labels for the items
- `k`: Number of top results to consider (optional)

#### Output:
A dictionary containing:
- `mAP`: Mean Average Precision
- `avg_relevant_items`: Average number of relevant items per query
- `median_relevant_items`: Median number of relevant items per query

#### Alignment Handling:
The function calculates alignment ratios between embeddings and labels, and aligns the labels accordingly using the `_calculate_ratios` and `_align_labels` helper functions.

### 2. Instance Retrieval

```python
def instance_retrieval(similarity_matrix: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
```

This function performs instance-level retrieval for both image-to-text and text-to-image directions.

#### Input:
- `similarity_matrix`: A precomputed similarity matrix of shape (num_images * 5, num_images)

#### Output:
A tuple of two dictionaries, one for image-to-text and one for text-to-image retrieval, each containing:
- `R@1`, `R@5`, `R@10`: Recall at 1, 5, and 10
- `MedianR`: Median rank
- `MeanR`: Mean rank

#### Alignment Handling:
The function assumes a specific alignment where each image has 5 corresponding captions. It uses the `check_similarity_matrix` helper function to ensure correct orientation of the similarity matrix.

## Metrics

1. **Mean Average Precision (mAP)**: Measures the average precision across all queries, considering the rank of all relevant items.

2. **Recall at K (R@K)**: The percentage of queries where a relevant item is found within the top K results.

3. **Median Rank (MedianR)**: The median position of the first relevant item in the ranked list of results.

4. **Mean Rank (MeanR)**: The average position of the first relevant item in the ranked list of results.

## Important Considerations

1. **Alignment**: The functions assume specific alignment ratios between images, captions, and labels. Ensure your data follows these assumptions or modify the alignment handling as needed.

2. **Similarity Matrix**: The instance retrieval function expects a similarity matrix where rows represent captions (5 per image) and columns represent images.

3. **Performance**: For large datasets, consider the memory requirements, especially for the similarity matrix computations.

## Usage Examples

### Category Retrieval

```python
from fico_itr.tasks import category_retrieval
from fico_itr.similarity import compute_similarity

# Assume img_embs, txt_embs, and labels are loaded
similarity_matrix = compute_similarity(img_embs, txt_embs)
results = category_retrieval(similarity_matrix, labels)
print(f"mAP: {results['mAP']:.4f}")
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

1. Currently assumes specific alignment ratios. Future versions may support more flexible alignment schemes.
2. Performance optimizations for very large datasets may be needed.
3. Support for additional retrieval metrics could be added in future releases.

Remember to refer to the alignment and similarity documentation for more details on how these aspects are handled within the tasks module.