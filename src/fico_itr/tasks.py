import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import warnings

def _calculate_ratios(n_images: int, n_captions: int, n_labels: int) -> Tuple[int, int]:
    """Calculate alignment ratios between embeddings and labels."""
    img_ratio = n_images // n_labels
    cap_ratio = n_captions // n_labels

    if n_images % n_labels != 0 or n_captions % n_labels != 0:
        import warnings
        warnings.warn(f"Non-integer ratios detected. Rounding down to img_ratio: {img_ratio}, cap_ratio: {cap_ratio}")

    return img_ratio, cap_ratio

def _align_labels(labels: np.ndarray, img_ratio: int) -> np.ndarray:
    """Align labels with embeddings."""
    return np.repeat(labels, img_ratio, axis=0)

def _compute_map(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    k: Optional[int] = None,
    captions_per_image: Optional[Union[int, List, np.ndarray]] = None
) -> float:
    n_queries, n_retrievals = similarity_matrix.shape
    n_labels = len(labels)

    if captions_per_image is not None:
        if isinstance(captions_per_image, (int, float)):
            cap_ratio = int(captions_per_image)
            
            if n_queries < n_retrievals:
                if n_queries != n_labels and n_queries != n_labels * cap_ratio:
                    raise ValueError(f"For i2t, expected {n_labels} or {n_labels * cap_ratio} image queries but got {n_queries}")
                if n_retrievals != n_labels * cap_ratio:
                    raise ValueError(
                        f"For i2t, expected {n_labels * cap_ratio} caption retrievals but got {n_retrievals}. "
                        f"Check if captions_per_image is set correctly for your model/dataset."
                    )
                
                if n_queries == n_labels * cap_ratio:
                    aligned_query_labels = _align_labels(labels, cap_ratio)
                else:
                    aligned_query_labels = labels
                aligned_retrieval_labels = _align_labels(labels, cap_ratio)
            else:
                if n_queries == n_retrievals and n_queries == n_labels * cap_ratio:
                    aligned_query_labels = _align_labels(labels, cap_ratio)
                    aligned_retrieval_labels = _align_labels(labels, cap_ratio)
                else:
                    if n_queries != n_labels * cap_ratio:
                        raise ValueError(
                        f"For t2i, expected {n_labels * cap_ratio} caption queries but got {n_queries}. "
                        f"This often happens with square matrices (e.g., vsrn/ucch models). "
                        f"Try setting captions_per_image={n_queries // n_labels} if applicable."
                    )
                    if n_retrievals != n_labels:
                        raise ValueError(f"For t2i, expected {n_labels} image retrievals but got {n_retrievals}")
                    
                    aligned_query_labels = _align_labels(labels, cap_ratio)
                    aligned_retrieval_labels = labels
        else:
            caption_to_image = np.array(captions_per_image)
            
            if n_queries < n_retrievals:
                aligned_query_labels = labels
                
                if labels.ndim == 1:
                    aligned_retrieval_labels = labels[caption_to_image]
                else:
                    aligned_retrieval_labels = labels[caption_to_image, :]
            else:
                if labels.ndim == 1:
                    aligned_query_labels = labels[caption_to_image]
                else:
                    aligned_query_labels = labels[caption_to_image, :]
                
                aligned_retrieval_labels = labels
    else:
        img_ratio, cap_ratio = _calculate_ratios(n_queries, n_retrievals, n_labels)
        aligned_query_labels = _align_labels(labels, img_ratio)
        aligned_retrieval_labels = _align_labels(labels, cap_ratio)

    k_val = k or n_retrievals
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    ap_scores = []

    for i in range(n_queries):
        current_query_labels = aligned_query_labels[i]
        indices = sorted_indices[i, :k_val]
        if np.any(indices >= len(aligned_retrieval_labels)):
            raise IndexError(f"Index {np.max(indices)} out of bounds for retrieval labels of length {len(aligned_retrieval_labels)}")
        retrieved_labels = aligned_retrieval_labels[indices]
        
        if labels.ndim == 1:
            relevant = retrieved_labels == current_query_labels
        else:
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
    sim2: Optional[np.ndarray] = None,
    captions_per_image: Union[str, int, List, np.ndarray, Dict] = 'auto'
) -> Tuple[float, float]:
    """
    Perform category-level retrieval for both image-to-text and text-to-image.

    Args:
        sim1 (np.ndarray): Precomputed similarity matrix (images × captions)
        labels (np.ndarray): Category labels
        k (int, optional): Number of top results to consider. If None, considers all.
        sim2 (np.ndarray, optional): Precomputed similarity matrix for text-to-image.
                                     If None, sim1.T will be used.
        captions_per_image: Caption distribution information. For uneven distributions,
                           explicit mapping is required to ensure ground truth relevance.

    Returns:
        Tuple[float, float]: (mAP_i2t, mAP_t2i) Retrieval results including mAP for both directions
    """
    n_images, n_captions = sim1.shape
    
    if n_images != n_captions and n_images > n_captions:
        warnings.warn(
            f"Transposing similarity matrix from {sim1.shape} to "
            f"{sim1.shape[::-1]} (expected images × captions)"
        )
        sim1 = sim1.T
        n_images, n_captions = sim1.shape
        if sim2 is not None:
            sim2 = sim2.T
    
    if n_images == n_captions and isinstance(captions_per_image, (int, float)) and captions_per_image > 1:
        if n_images % captions_per_image != 0:
            raise ValueError(
                f"Square matrix size {n_images}×{n_captions} is not divisible by "
                f"captions_per_image={captions_per_image}. For duplicated image matrices "
                f"(e.g., vsrn/ucch), use the original ratio (5 for COCO/Flickr30k)."
            )
        n_unique_images = n_images // int(captions_per_image)
        caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
            captions_per_image, n_unique_images, n_unique_images * int(captions_per_image)
        )
    else:
        caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
            captions_per_image, n_images, n_captions
        )
    
    if n_images == n_captions and isinstance(captions_per_image, (int, float)) and captions_per_image > 1:
        mAP_i2t = _compute_map(sim1, labels, k, int(captions_per_image))
        
        if sim2 is None:
            sim2 = sim1.T
        
        mAP_t2i = _compute_map(sim2, labels, k, int(captions_per_image))
    else:
        param_to_pass = cpi_value if is_uniform else caption_mapping
        mAP_i2t = _compute_map(sim1, labels, k, param_to_pass)
        
        if sim2 is None:
            sim2 = sim1.T
        
        mAP_t2i = _compute_map(sim2, labels, k, param_to_pass)

    return mAP_i2t, mAP_t2i

def instance_i2t(
    similarity_matrix: np.ndarray,
    captions_per_image: Union[str, int, List, np.ndarray, Dict] = 'auto'
) -> Dict[str, float]:
    """
    Perform image-to-text retrieval.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix (images × captions)
        captions_per_image: Caption distribution information
    
    Returns:
        Dict[str, float]: Retrieval results including R@1, R@5, R@10, MedianR, and MeanR
    """
    n_rows, n_cols = similarity_matrix.shape
    
    if n_rows == n_cols:
        if isinstance(captions_per_image, (int, float)) and captions_per_image > 1:
            if n_rows % captions_per_image != 0:
                raise ValueError(
                    f"Square matrix size {n_rows}×{n_cols} is not divisible by "
                    f"captions_per_image={captions_per_image}. For duplicated image matrices "
                    f"(e.g., vsrn/ucch), use the original ratio (5 for COCO/Flickr30k)."
                )
            n_unique_images = n_rows // int(captions_per_image)
            caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
                captions_per_image, n_unique_images, n_unique_images * int(captions_per_image)
            )
        else:
            caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
                captions_per_image, n_rows, n_cols
            )
        
        if is_uniform:
            npts = n_rows // cpi_value
            im_dupe = cpi_value
            txt_per_im = cpi_value
            
            ranks = np.zeros(npts)
            
            for index in range(npts):
                d = similarity_matrix[im_dupe * index]
                inds = np.argsort(d)[::-1]
                
                rank = 1e20
                for i in range(txt_per_im * index, txt_per_im * index + txt_per_im):
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank
        else:
            raise ValueError(
                "Square matrix with non-uniform caption distribution is ambiguous. "
                "Please provide rectangular matrix (images × captions) for uneven distributions. "
                "Alternatively, if this is a duplicated image matrix (e.g., vsrn/ucch), "
                "use captions_per_image=5 (or appropriate ratio) instead of a mapping."
            )
    else:
        if n_rows > n_cols:
            warnings.warn(
                f"Transposing similarity matrix from {similarity_matrix.shape} to "
                f"{similarity_matrix.shape[::-1]} (expected images × captions)"
            )
            similarity_matrix = similarity_matrix.T
            n_rows, n_cols = n_cols, n_rows
        
        npts = n_rows
        n_captions = n_cols
        
        caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
            captions_per_image, npts, n_captions
        )
        
        ranks = np.zeros(npts)
        
        for img_idx in range(npts):
            d = similarity_matrix[img_idx]
            inds = np.argsort(d)[::-1]
            
            if is_uniform:
                target_captions = list(range(img_idx * cpi_value, (img_idx + 1) * cpi_value))
            else:
                target_captions = np.where(caption_mapping == img_idx)[0].tolist()
            
            rank = 1e20
            for cap_idx in target_captions:
                if cap_idx < n_captions:
                    tmp = np.where(inds == cap_idx)[0]
                    if len(tmp) > 0:
                        if tmp[0] < rank:
                            rank = tmp[0]
            
            ranks[img_idx] = rank

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedianR": medr, "MeanR": meanr}

def instance_t2i(
    similarity_matrix: np.ndarray,
    captions_per_image: Union[str, int, List, np.ndarray, Dict] = 'auto'
) -> Dict[str, float]:
    """
    Perform text-to-image retrieval.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix
        captions_per_image: Caption distribution information
    
    Returns:
        Dict[str, float]: Retrieval results including R@1, R@5, R@10, MedianR, and MeanR
    """
    n_rows, n_cols = similarity_matrix.shape
    
    if n_rows == n_cols:
        if isinstance(captions_per_image, (int, float)) and captions_per_image > 1:
            if n_rows % captions_per_image != 0:
                raise ValueError(
                    f"Square matrix size {n_rows}×{n_cols} is not divisible by "
                    f"captions_per_image={captions_per_image}. For duplicated image matrices "
                    f"(e.g., vsrn/ucch), use the original ratio (5 for COCO/Flickr30k)."
                )
            n_unique_images = n_rows // int(captions_per_image)
            caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
                captions_per_image, n_unique_images, n_unique_images * int(captions_per_image)
            )
        else:
            caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
                captions_per_image, n_rows, n_cols
            )
        
        if is_uniform:
            n_unique_images = n_rows // cpi_value
            unique_img_indices = np.arange(0, n_rows, cpi_value)
            
            ranks = np.zeros(n_rows)
            
            for cap_idx in range(n_rows):
                d = similarity_matrix[unique_img_indices, cap_idx]
                inds = np.argsort(d)[::-1]
                
                target_unique_img = cap_idx // cpi_value
                
                rank = np.where(inds == target_unique_img)[0][0]
                ranks[cap_idx] = rank
        else:
            raise ValueError(
                "Square matrix with non-uniform caption distribution is ambiguous. "
                "For duplicated image matrices (e.g., vsrn/ucch), use captions_per_image=5. "
                "For true non-uniform distributions, provide rectangular matrix instead."
            )
    else:
        if n_rows > n_cols:
            warnings.warn(
                f"Transposing similarity matrix from {similarity_matrix.shape} to "
                f"{similarity_matrix.shape[::-1]} (expected images × captions)"
            )
            similarity_matrix = similarity_matrix.T
            n_rows, n_cols = n_cols, n_rows
        
        n_images = n_rows
        n_captions = n_cols
        
        caption_mapping, is_uniform, cpi_value = _create_caption_mapping(
            captions_per_image, n_images, n_captions
        )
        
        similarity_matrix = similarity_matrix.T
        
        ranks = np.zeros(n_captions)
        
        for cap_idx in range(n_captions):
            d = similarity_matrix[cap_idx, :]
            inds = np.argsort(d)[::-1]
            
            if is_uniform:
                correct_img_idx = cap_idx // cpi_value
            else:
                correct_img_idx = caption_mapping[cap_idx]
            
            rank = np.where(inds == correct_img_idx)[0][0]
            ranks[cap_idx] = rank

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedianR": medr, "MeanR": meanr}

def _create_caption_mapping(captions_per_image, n_images, n_captions):
    """
    Create caption-to-image mapping from various input formats.
    
    Args:
        captions_per_image: Can be:
            - 'auto': Try to infer from matrix shape
            - int: Fixed number of captions per image
            - list/array of ints: Caption count for each image
            - list/array where index=caption_id, value=image_id
            - dict: {caption_id: image_id} mapping
        n_images: Number of images in the dataset
        n_captions: Number of captions in the dataset
    
    Returns:
        tuple: (caption_to_image_array, is_uniform, captions_per_image_value)
            - caption_to_image_array: None if uniform distribution, else numpy array
            - is_uniform: True if all images have same caption count
            - captions_per_image_value: int if uniform, else None
    """
    if captions_per_image == 'auto':
        if n_captions % n_images == 0:
            captions_per_image_value = n_captions // n_images
            return None, True, captions_per_image_value
        else:
            raise ValueError(
                f"Uneven caption distribution detected: {n_captions} captions "
                f"for {n_images} images is not evenly divisible. "
                f"For datasets with uneven distributions (like COCO with 25010 captions), "
                f"you MUST provide explicit caption-to-image mapping. "
                f"Options: 1) Use captions_per_image as a list/array of caption indices, "
                f"2) For square matrices, try captions_per_image=5 (common for COCO/Flickr30k)."
            )
    
    elif isinstance(captions_per_image, int):
        expected_captions = n_images * captions_per_image
        if expected_captions != n_captions:
            raise ValueError(
                f"Mismatch: {n_images} images × {captions_per_image} captions/image "
                f"= {expected_captions} expected captions, but got {n_captions}. "
                f"For square matrices (like vsrn/ucch), images might be duplicated - "
                f"try using the original caption ratio (usually 5 for COCO/Flickr30k)."
            )
        return None, True, captions_per_image
    
    elif isinstance(captions_per_image, dict):
        mapping = np.zeros(n_captions, dtype=np.int32)
        for cap_id, img_id in captions_per_image.items():
            if cap_id >= n_captions:
                raise ValueError(f"Caption ID {cap_id} exceeds number of captions {n_captions}")
            if img_id >= n_images:
                raise ValueError(f"Image ID {img_id} exceeds number of images {n_images}")
            mapping[cap_id] = img_id
        return mapping, False, None
    
    elif isinstance(captions_per_image, (list, np.ndarray)):
        captions_array = np.array(captions_per_image)
        
        if len(captions_array) == n_images:
            if captions_array.sum() != n_captions:
                raise ValueError(
                    f"Sum of caption counts ({captions_array.sum()}) "
                    f"doesn't match number of captions ({n_captions})"
                )
            mapping = np.zeros(n_captions, dtype=np.int32)
            idx = 0
            for img_id, count in enumerate(captions_array):
                mapping[idx:idx+count] = img_id
                idx += count
            return mapping, False, None
        
        elif len(captions_array) == n_captions:
            return captions_array.astype(np.int32), False, None
        
        else:
            raise ValueError(
                f"Array length {len(captions_array)} doesn't match "
                f"n_images ({n_images}) or n_captions ({n_captions})"
            )
    
    else:
        raise ValueError(
            f"Unsupported captions_per_image type: {type(captions_per_image)}. "
            f"Expected 'auto', int, list, array, or dict."
        )


def instance_retrieval(
    similarity_matrix: np.ndarray, 
    t2i_sim: Optional[np.ndarray] = None,
    captions_per_image: Union[str, int, List, np.ndarray, Dict] = 'auto'
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Perform instance-level retrieval for both image-to-text and text-to-image.

    Args:
        similarity_matrix (np.ndarray): Precomputed similarity matrix
        t2i_sim (np.ndarray, optional): Separate matrix for t2i if different from i2t
        captions_per_image: Caption distribution information. Can be:
            - 'auto': Infer from matrix shape (default)
            - int: Fixed number of captions per image (e.g., 5)
            - list/array: Caption counts per image or caption-to-image mapping
            - dict: Explicit {caption_id: image_id} mapping

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Results for i2t and t2i retrieval
    """

    if t2i_sim is not None:
        i2t_results = instance_i2t(similarity_matrix, captions_per_image)
        t2i_results = instance_t2i(t2i_sim, captions_per_image)
    else:
        i2t_results = instance_i2t(similarity_matrix, captions_per_image)
        t2i_results = instance_t2i(similarity_matrix, captions_per_image)
    
    return i2t_results, t2i_results