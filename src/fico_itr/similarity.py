__all__ = ['compute_similarity']

import numpy as np
from typing import Union, Literal

def compute_similarity(
    x: np.ndarray,
    y: np.ndarray,
    measure: Literal['cosine', 'euclidean', 'hamming', 'inner_product'] = 'cosine'
) -> np.ndarray:
    """
    Compute similarity matrix between two sets of vectors.

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
    """
    if measure == 'cosine':
        # Normalize the vectors
        x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
        return np.dot(x_norm, y_norm.T)

    elif measure == 'euclidean':
        # Compute pairwise euclidean distances
        return -np.sqrt(np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2))

    elif measure == 'hamming':
        # Assuming binary vectors
        return -np.sum(x[:, np.newaxis, :] != y[np.newaxis, :, :], axis=2) / x.shape[1]

    elif measure == 'inner_product':
        return np.dot(x, y.T)

    else:
        raise ValueError(f"Invalid similarity measure: {measure}")