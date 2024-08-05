__all__ = ['category_retrieval', 'instance_retrieval_i2t', 'instance_retrieval_t2i']

import numpy as np
from typing import Tuple, Optional, Union

def category_retrieval(
    similarity_matrix: np.ndarray,
    query_labels: np.ndarray,
    retrieval_labels: np.ndarray,
    k: Optional[int] = None
) -> dict:
    """
    Evaluate category-level retrieval performance using mean Average Precision (mAP).

    Args:
        similarity_matrix (np.ndarray): Pre-computed similarity matrix of shape (n_queries, n_retrievals)
        query_labels (np.ndarray): Query labels with shape (n_queries, n_categories)
        retrieval_labels (np.ndarray): Retrieval labels with shape (n_retrievals, n_categories)
        k (int, optional): Number of top-k retrievals to consider for mAP calculation.
            If None, all retrievals are considered. Default is None.

    Returns:
        dict: A dictionary containing evaluation metrics:
            - 'mAP': Mean Average Precision score
            - 'avg_relevant_items': Average number of relevant items per query
            - 'median_relevant_items': Median number of relevant items per query
    """
    n_queries, n_retrievals = similarity_matrix.shape

    if n_queries != query_labels.shape[0] or n_retrievals != retrieval_labels.shape[0]:
        raise ValueError("Mismatch between similarity matrix shape and label counts")

    k = k or n_retrievals
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    ap_scores = []
    relevant_num_list = []

    for i in range(n_queries):
        current_query_labels = query_labels[i]
        retrieved_labels = retrieval_labels[sorted_indices[i, :k]]
        
        relevant = np.any(np.logical_and(retrieved_labels, current_query_labels), axis=1)
        relevant_indices = np.where(relevant)[0]
        relevant_num_list.append(len(relevant_indices))

        if len(relevant_indices) > 0:
            precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
            ap = np.mean(precision_at_k[relevant])
        else:
            ap = 0.0
        ap_scores.append(ap)
    
    mAP = np.mean(ap_scores)
    avg_relevant_num = np.mean(relevant_num_list)
    median_relevant_num = np.median(relevant_num_list)

    return {
        'mAP': mAP,
        'avg_relevant_items': avg_relevant_num,
        'median_relevant_items': median_relevant_num
    }

def instance_retrieval_i2t(
    similarity_matrix: np.ndarray,
    n_captions_per_image: int = 5,
    return_ranks: bool = False
) -> Union[Tuple[float, float, float, float, float], Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]]]:
    """
    Evaluate instance-level image-to-text retrieval performance.

    Args:
        similarity_matrix (np.ndarray): Pre-computed similarity matrix of shape (n_images, n_captions)
        n_captions_per_image (int): Number of captions per image. Default is 5.
        return_ranks (bool): If True, return ranks and top1 indices along with metrics. Default is False.

    Returns:
        If return_ranks is False:
            Tuple[float, float, float, float, float]: (R@1, R@5, R@10, median rank, mean rank)
        If return_ranks is True:
            Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]]:
                ((R@1, R@5, R@10, median rank, mean rank), (ranks, top1))
    """
    n_images = similarity_matrix.shape[0]
    n_captions = similarity_matrix.shape[1]
    
    if n_images * n_captions_per_image != n_captions:
        raise ValueError("Mismatch between number of images and captions")

    ranks = np.zeros(n_images)
    top1 = np.zeros(n_images)

    for i in range(n_images):
        d = similarity_matrix[i]
        inds = np.argsort(d)[::-1]
        rank = min(np.where(inds == j)[0][0] for j in range(i*n_captions_per_image, (i+1)*n_captions_per_image))
        
        ranks[i] = rank
        top1[i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def instance_retrieval_t2i(
    similarity_matrix: np.ndarray,
    n_captions_per_image: int = 5,
    return_ranks: bool = False
) -> Union[Tuple[float, float, float, float, float], Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]]]:
    """
    Evaluate instance-level text-to-image retrieval performance.

    Args:
        similarity_matrix (np.ndarray): Pre-computed similarity matrix of shape (n_captions, n_images)
        n_captions_per_image (int): Number of captions per image. Default is 5.
        return_ranks (bool): If True, return ranks and top1 indices along with metrics. Default is False.

    Returns:
        If return_ranks is False:
            Tuple[float, float, float, float, float]: (R@1, R@5, R@10, median rank, mean rank)
        If return_ranks is True:
            Tuple[Tuple[float, float, float, float, float], Tuple[np.ndarray, np.ndarray]]:
                ((R@1, R@5, R@10, median rank, mean rank), (ranks, top1))
    """
    n_captions, n_images = similarity_matrix.shape
    
    if n_images * n_captions_per_image != n_captions:
        raise ValueError("Mismatch between number of images and captions")

    n_queries = n_captions
    ranks = np.zeros(n_queries)
    top1 = np.zeros(n_queries)

    for i in range(n_queries):
        d = similarity_matrix[i]
        inds = np.argsort(d)[::-1]
        ranks[i] = np.where(inds == i // n_captions_per_image)[0][0]
        top1[i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)