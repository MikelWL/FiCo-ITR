import numpy as np
from typing import Tuple, Optional, Union, List, Dict
from .similarity import compute_similarity

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

def category_retrieval(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    k: Optional[int] = None
) -> dict:
    """
    Perform category-level retrieval.

    Args:
        similarity_matrix (np.ndarray): Precomputed similarity matrix
        labels (np.ndarray): Category labels
        k (int, optional): Number of top results to consider. If None, considers all.

    Returns:
        dict: Retrieval results including mAP and other metrics
    """
    n_queries, n_retrievals = similarity_matrix.shape
    n_labels = len(labels)

    img_ratio, cap_ratio = _calculate_ratios(n_queries, n_retrievals, n_labels)
    
    aligned_query_labels = _align_labels(labels, img_ratio)
    aligned_retrieval_labels = _align_labels(labels, cap_ratio)

    k = k or n_retrievals
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    ap_scores = []
    relevant_num_list = []

    for i in range(n_queries):
        current_query_labels = aligned_query_labels[i]
        retrieved_labels = aligned_retrieval_labels[sorted_indices[i, :k]]
        
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

def check_similarity_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Check and potentially reshape the similarity matrix to ensure correct orientation.
    
    Args:
        similarity_matrix (np.ndarray): Input similarity matrix
    
    Returns:
        np.ndarray: Correctly shaped similarity matrix
    """
    print(f"Original similarity matrix shape: {similarity_matrix.shape}")
    
    if similarity_matrix.shape[0] % 5 != 0:
        print(f"Warning: Similarity matrix first dimension ({similarity_matrix.shape[0]}) is not divisible by 5")
    
    if similarity_matrix.shape[0] < similarity_matrix.shape[1]:
        print("Transposing similarity matrix")
        similarity_matrix = similarity_matrix.T
    
    print(f"Final similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix

def image_to_text_retrieval(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Perform image-to-text retrieval.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix of shape (num_images * 5, num_images)
    
    Returns:
        Dict[str, float]: Retrieval results including R@1, R@5, R@10, MedianR, and MeanR
    """
    similarity_matrix = check_similarity_matrix(similarity_matrix)
    npts = similarity_matrix.shape[0] // 5
    ranks = np.zeros(npts)
    
    for index in range(npts):
        # Get query image
        d = similarity_matrix[5 * index]
        inds = np.argsort(d)[::-1]
        
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
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

def text_to_image_retrieval(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Perform text-to-image retrieval.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix of shape (num_captions, num_images)
    
    Returns:
        Dict[str, float]: Retrieval results including R@1, R@5, R@10, MedianR, and MeanR
    """
    similarity_matrix = check_similarity_matrix(similarity_matrix)
    print(f"Input similarity matrix shape: {similarity_matrix.shape}")
    
    n_captions, n_images = similarity_matrix.shape
    npts = n_captions // 5

    ranks = np.zeros(n_captions)

    for caption_index in range(0, n_captions, 5):
        for i in range(5):
            d = similarity_matrix[caption_index + i]
            inds = np.argsort(d)[::-1]
            
            # Find the rank of the correct image
            correct_image_index = caption_index // 5
            rank = np.where(inds == correct_image_index)[0][0]
            ranks[caption_index + i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    print(f"Computed ranks shape: {ranks.shape}")
    print(f"Ranks statistics - Min: {ranks.min()}, Max: {ranks.max()}, Mean: {ranks.mean()}")

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedianR": medr, "MeanR": meanr}

def text_to_image_retrieval_old(similarity_matrix: np.ndarray) -> Dict[str, float]:
    """
    Perform text-to-image retrieval.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix of shape (num_captions, num_images)
    
    Returns:
        Dict[str, float]: Retrieval results including R@1, R@5, R@10, MedianR, and MeanR
    """
    print(f"Input similarity matrix shape: {similarity_matrix.shape}")
    
    n_captions, n_images = similarity_matrix.shape
    n_unique_images = n_images // 5 if n_images > n_captions else n_images

    print(f"Detected {n_unique_images} unique images")

    # If images are repeated, we need to use only unique images
    if n_images > n_captions:
        similarity_matrix = similarity_matrix[:, ::5]
        print(f"Adjusted similarity matrix shape: {similarity_matrix.shape}")

    ranks = np.zeros(n_captions)

    for caption_index in range(0, n_captions, 5):
        # Get query captions (5 captions per image)
        caption_scores = similarity_matrix[caption_index:caption_index+5]
        
        if caption_scores.shape[0] == 0:
            print(f"Warning: No caption scores found for index {caption_index}")
            continue

        # Compute ranks for each caption
        for i in range(min(5, caption_scores.shape[0])):
            inds = np.argsort(caption_scores[i])[::-1]
            ranks[caption_index + i] = np.where(inds == caption_index // 5)[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    print(f"Computed ranks shape: {ranks.shape}")
    print(f"Ranks statistics - Min: {ranks.min()}, Max: {ranks.max()}, Mean: {ranks.mean()}")

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedianR": medr, "MeanR": meanr}

def instance_retrieval(similarity_matrix: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Perform instance-level retrieval for both image-to-text and text-to-image.

    Args:
        similarity_matrix (np.ndarray): Precomputed similarity matrix of shape (num_images * 5, num_images)

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Results for image-to-text and text-to-image retrieval
    """
    i2t_results = image_to_text_retrieval(similarity_matrix)
    t2i_results = text_to_image_retrieval(similarity_matrix)
    
    return i2t_results, t2i_results