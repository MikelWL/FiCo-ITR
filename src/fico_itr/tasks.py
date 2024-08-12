import numpy as np
from typing import Tuple, Optional, Union, List
from .similarity import compute_similarity

import numpy as np
from typing import Tuple, Optional

class ImageTextRetrieval:
    """
    Handles image-text retrieval tasks with automatic alignment detection.

    This class supports various alignment scenarios between image and caption embeddings,
    including balanced sets, unbalanced sets, and pre-aligned data.

    Attributes:
        image_embeddings (np.ndarray): Image embedding vectors.
        caption_embeddings (np.ndarray): Caption embedding vectors.
        labels (np.ndarray, optional): Labels for category retrieval tasks.
        img_ratio (int): Ratio of image embeddings to unique images.
        cap_ratio (int): Ratio of caption embeddings to unique images.

    Args:
        image_embeddings (np.ndarray): Image embedding vectors.
        caption_embeddings (np.ndarray): Caption embedding vectors.
        labels (np.ndarray, optional): Labels for category retrieval tasks.

    Note:
        For detailed information on alignment handling, refer to docs/alignment.md.
    """
    def __init__(self, image_embeddings: np.ndarray, caption_embeddings: np.ndarray, labels: Optional[np.ndarray] = None):
        self.image_embeddings = image_embeddings
        self.caption_embeddings = caption_embeddings
        self.labels = labels
        self.aligned_labels = None
        self.img_indices = np.arange(len(image_embeddings))
        self.cap_indices = np.arange(len(caption_embeddings))

        if self.labels is not None:
            self.img_ratio, self.cap_ratio = self._calculate_ratios()

    def prepare_for_category_retrieval(self):
        """
        Prepare the instance for category retrieval tasks.

        This method calculates alignment ratios and aligns labels with embeddings.

        Raises:
            ValueError: If labels are not provided for category retrieval.
        """
        if self.labels is None:
            raise ValueError("Labels are required for category retrieval tasks.")
        
        if self.img_ratio is None or self.cap_ratio is None:
            self.img_ratio, self.cap_ratio = self._calculate_ratios()
        
        self.aligned_labels = self._align_labels()

    def _calculate_ratios(self) -> Tuple[int, int]:
        """
        Calculate alignment ratios between embeddings and labels.

        Returns:
            Tuple[int, int]: Image ratio and caption ratio.

        Raises:
            NotImplementedError: If non-integer ratios are detected.
        """
        n_images, n_captions, n_labels = len(self.image_embeddings), len(self.caption_embeddings), len(self.labels)
        
        img_ratio = n_images // n_labels
        cap_ratio = n_captions // n_labels

        if n_images % n_labels != 0 or n_captions % n_labels != 0:
            print(f"Warning: Non-integer ratios detected. img_ratio: {n_images/n_labels}, cap_ratio: {n_captions/n_labels}")
            print(f"Rounding down to img_ratio: {img_ratio}, cap_ratio: {cap_ratio}")

        return img_ratio, cap_ratio


    def _align_labels(self) -> np.ndarray:
        return np.repeat(self.labels, self.img_ratio, axis=0)

    def compute_similarity(self, measure: str = 'cosine') -> np.ndarray:
        from .similarity import compute_similarity
        return compute_similarity(self.image_embeddings, self.caption_embeddings, measure)

    def get_alignment_info(self) -> dict:
        """
        Get information about the current alignment.

        Returns:
            dict: A dictionary containing alignment information.
        """
        info = {
            "n_image_embeddings": len(self.image_embeddings),
            "n_caption_embeddings": len(self.caption_embeddings),
        }
        if self.labels is not None:
            info.update({
                "n_labels": len(self.labels),
                "image_ratio": self.img_ratio,
                "caption_ratio": self.cap_ratio,
            })
        return info

def category_retrieval(
    retriever: ImageTextRetrieval,
    k: Optional[int] = None
) -> dict:
    retriever.prepare_for_category_retrieval()
    similarity_matrix = retriever.compute_similarity()
    n_queries, n_retrievals = similarity_matrix.shape

    alignment_info = retriever.get_alignment_info()
    print(f"Alignment info: {alignment_info}")

    aligned_query_labels = retriever.aligned_labels
    aligned_retrieval_labels = np.repeat(retriever.labels, retriever.cap_ratio, axis=0)

    if n_queries != len(aligned_query_labels) or n_retrievals != len(aligned_retrieval_labels):
        raise ValueError(f"Mismatch between similarity matrix shape {similarity_matrix.shape} and aligned label counts ({len(aligned_query_labels)}, {len(aligned_retrieval_labels)})")

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

def instance_retrieval_i2t(retriever: ImageTextRetrieval) -> Tuple[float, float, float, float, float]:
    similarity_matrix = retriever.compute_similarity()
    n_images = len(retriever.image_embeddings)
    n_captions = len(retriever.caption_embeddings)
    
    ranks = np.zeros(n_images)
    for i in range(n_images):
        inds = np.argsort(similarity_matrix[i])[::-1]
        ranks[i] = np.where(inds == i)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr)

def instance_retrieval_t2i(retriever: ImageTextRetrieval) -> Tuple[float, float, float, float, float]:
    similarity_matrix = retriever.compute_similarity()
    n_images = len(retriever.image_embeddings)
    n_captions = len(retriever.caption_embeddings)
    
    ranks = np.zeros(n_captions)
    for i in range(n_captions):
        inds = np.argsort(similarity_matrix[:, i])[::-1]
        ranks[i] = np.where(inds == i % n_images)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr)