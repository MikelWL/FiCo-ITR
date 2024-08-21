"""FiCo-ITR: Fine-grained and Coarse-grained Image-Text Retrieval Library"""

from .tasks import category_retrieval, instance_retrieval
from .similarity import compute_similarity

__all__ = ['category_retrieval', 'instance_retrieval', 'compute_similarity']

__version__ = "0.1.0"