"""FiCo-ITR: Fine-grained and Coarse-grained Image-Text Retrieval Library"""

from .tasks import ImageTextRetrieval, category_retrieval, instance_retrieval_i2t, instance_retrieval_t2i
from .similarity import compute_similarity

__all__ = ['ImageTextRetrieval', 'category_retrieval', 'instance_retrieval_i2t', 'instance_retrieval_t2i', 'compute_similarity']

__version__ = "0.1.0"