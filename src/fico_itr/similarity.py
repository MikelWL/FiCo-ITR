__all__ = ['compute_similarity']

import numpy as np
# TODO: Uncomment these as they were added in Python 3.8
# from typing import Union, Literal
from typing import Union
from typing_extensions import Literal

def compute_similarity(
    x: np.ndarray,
    y: np.ndarray,
    measure: Literal['cosine', 'euclidean', 'hamming', 'inner_product'] = 'cosine'
) -> np.ndarray:
    """
    Compute similarity matrix between two sets of vectors.

    This function calculates the similarity between each pair of vectors in x and y
    using the specified measure. It's optimized for efficiency and numerical stability.

    Args:
        x (np.ndarray): First set of vectors, shape (n, d)
        y (np.ndarray): Second set of vectors, shape (m, d)
        measure (str): Similarity measure to use.
                       Options: 'cosine', 'euclidean', 'hamming', 'inner_product'
                       Default: 'cosine'

    Returns:
        np.ndarray: Similarity matrix of shape (n, m)

    Raises:
        ValueError: If an invalid similarity measure is specified

    Notes:
        - Cosine similarity: Range [-1, 1], higher values indicate more similarity
        - Euclidean similarity: Transformed to similarity, range (0, 1], higher values indicate more similarity
        - Hamming similarity: Range [-1, 0], higher values (closer to 0) indicate more similarity
        - Inner product: No fixed range, higher values indicate more similarity

    Performance considerations:
        - This implementation, although optimised, may not be the fastest for very large datasets.
        - For very large datasets, consider using a more efficient implementation based on FAISS or other such similarity search-centric libraries.
    """
    if measure == 'cosine':
        x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
        return np.dot(x_norm, y_norm.T)
    
    elif measure == 'euclidean':
        xx = np.sum(x**2, axis=1)[:, np.newaxis]
        yy = np.sum(y**2, axis=1)[np.newaxis, :]
        distances = xx + yy - 2 * np.dot(x, y.T)
        return 1 / (1 + np.sqrt(np.maximum(distances, 0)) + 1e-8)
    
    elif measure == 'hamming':
        return -np.sum(x[:, np.newaxis, :] != y[np.newaxis, :, :], axis=2) / x.shape[1]
    
    elif measure == 'inner_product':
        return np.dot(x, y.T)
    
    else:
        raise ValueError(f"Invalid similarity measure: {measure}")