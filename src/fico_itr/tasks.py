import numpy as np
from typing import Tuple, Optional, Union, List, Dict

def _calculate_ratios(n_images: int, n_captions: int, n_labels: int) -> Tuple[int, int]:
    """
    Calculate alignment ratios between embeddings and labels.
    """
    img_ratio = n_images // n_labels
    cap_ratio = n_captions // n_labels

    if n_images % n_labels != 0 or n_captions % n_labels != 0:
        print(f"Warning: Non-integer ratios detected. img_ratio: {n_images/n_labels}, cap_ratio: {n_captions/n_labels}")
        print(f"Rounding down to img_ratio: {img_ratio}, cap_ratio: {cap_ratio}")

    return img_ratio, cap_ratio

def _align_labels(labels: np.ndarray, img_ratio: int) -> np.ndarray:
    """
    Align labels with embeddings.
    """
    return np.repeat(labels, img_ratio, axis=0)

def _compute_map(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    k: Optional[int] = None
) -> float:
    n_queries, n_retrievals = similarity_matrix.shape
    n_labels = len(labels)

    img_ratio, cap_ratio = _calculate_ratios(n_queries, n_retrievals, n_labels)
    
    aligned_query_labels = _align_labels(labels, img_ratio)
    aligned_retrieval_labels = _align_labels(labels, cap_ratio)

    k_val = k or n_retrievals
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    ap_scores = []

    for i in range(n_queries):
        current_query_labels = aligned_query_labels[i]
        retrieved_labels = aligned_retrieval_labels[sorted_indices[i, :k_val]]
        
        relevant = np.any(np.logical_and(retrieved_labels, current_query_labels), axis=1)
        relevant_indices = np.where(relevant)[0]

        if len(relevant_indices) > 0:
            precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
            ap = np.mean(precision_at_k[relevant])
        else:
            ap = 0.0
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

def category_retrieval(
    sim1: np.ndarray,
    labels: np.ndarray,
    k: Optional[int] = None,
    sim2: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Perform category-level retrieval for both image-to-text and text-to-image.

    Args:
        sim1 (np.ndarray): Precomputed similarity matrix for image-to-text
        labels (np.ndarray): Category labels
        k (int, optional): Number of top results to consider. If None, considers all.
        sim2 (np.ndarray, optional): Precomputed similarity matrix for text-to-image.
                                     If None, sim1.T will be used.

    Returns:
        Tuple[float, float]: (mAP_i2t, mAP_t2i) Retrieval results including mAP for both directions
    """
    mAP_i2t = _compute_map(sim1, labels, k)
    
    if sim2 is None:
        sim2 = sim1.T
    
    mAP_t2i = _compute_map(sim2, labels, k)

    return mAP_i2t, mAP_t2i

def instance_i2t(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Perform image-to-text retrieval.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix of shape (num_images * 5, num_images)
    
    Returns:
        Dict[str, float]: Retrieval results including R@1, R@5, R@10, MedianR, and MeanR
    """
    # similarity_matrix = check_similarity_matrix(similarity_matrix)
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        npts = similarity_matrix.shape[0] // 5
        im_dupe = 5
    else:
        npts = similarity_matrix.shape[0]
        im_dupe = 1
    
    txt_per_im = 5

    ranks = np.zeros(npts)
    
    for index in range(npts):
        # Get query image
        d = similarity_matrix[im_dupe * index]
        inds = np.argsort(d)[::-1]
        
        # Score
        rank = 1e20
        for i in range(txt_per_im * index, txt_per_im * index + txt_per_im, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedianR": medr, "MeanR": meanr}

def instance_t2i(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Perform text-to-image retrieval.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix of shape (num_captions, num_images)
    
    Returns:
        Dict[str, float]: Retrieval results including R@1, R@5, R@10, MedianR, and MeanR
    """
    # similarity_matrix = check_similarity_matrix(similarity_matrix)
    # print(f"Input similarity matrix shape: {similarity_matrix.shape}")
    
    if similarity_matrix.shape[0] == similarity_matrix.shape[1]:
        unique_image_similarities = similarity_matrix[:, ::5]
        txt_per_im = 5
    else:
        if similarity_matrix.shape[0] < similarity_matrix.shape[1]:
            similarity_matrix = similarity_matrix.T
        unique_image_similarities = similarity_matrix
        txt_per_im = 5

    n_captions = similarity_matrix.shape[0]

    ranks = np.zeros(n_captions)

    for caption_index in range(n_captions):
        d = unique_image_similarities[caption_index]
        inds = np.argsort(d)[::-1]
        
        # Find the rank of the correct image
        correct_image_index = caption_index // txt_per_im
        rank = np.where(inds == correct_image_index)[0][0]
        ranks[caption_index] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    print(f"Computed ranks shape: {ranks.shape}")
    print(f"Ranks statistics - Min: {ranks.min()}, Max: {ranks.max()}, Mean: {ranks.mean()}")

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedianR": medr, "MeanR": meanr}

def instance_retrieval(similarity_matrix: np.ndarray, t2i_sim = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Perform instance-level retrieval for both image-to-text and text-to-image.

    Args:
        similarity_matrix (np.ndarray): Precomputed similarity matrix of shape (num_images * 5, num_images)

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Results for image-to-text and text-to-image retrieval
    """
    if t2i_sim is not None:
        i2t_results = instance_i2t(similarity_matrix)
        t2i_results = instance_t2i(t2i_sim)
    else:
        i2t_results = instance_i2t(similarity_matrix)
        t2i_results = instance_t2i(similarity_matrix)
    
    return i2t_results, t2i_results