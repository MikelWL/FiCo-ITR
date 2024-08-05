# FiCo-ITR: Fine-grained and Coarse-grained Image-Text Retrieval Library

**Note: This is a work in progress minimal working version. The full implementation will be released upon acceptance of the accompanying paper.**

FiCo-ITR is a Python library designed to facilitate unified evaluation of fine-grained and coarse-grained image-text retrieval models. It provides tools for computing similarity matrices and performing various retrieval tasks.

## Installation

You can install FiCo-ITR using pip:

```bash
pip install fico_itr
```

## Dependencies

FiCo-ITR requires the following Python libraries:

- numpy >= 1.19.0
- h5py >= 3.1.0

These dependencies will be automatically installed when you install FiCo-ITR using pip.

## Quick Start

Here's a simple example of how to use FiCo-ITR:

```python
import numpy as np
from fico_itr import compute_similarity, image_to_text_retrieval

# Load your image and text embeddings
image_embeddings = np.load('path_to_image_embeddings.npy')
text_embeddings = np.load('path_to_text_embeddings.npy')

# Compute similarity matrix
similarity_matrix = compute_similarity(image_embeddings, text_embeddings, measure='cosine')

# Perform image-to-text retrieval
results = image_to_text_retrieval(similarity_matrix)

print(f"R@1: {results[0]:.2f}%, R@5: {results[1]:.2f}%, R@10: {results[2]:.2f}%")
```

## Features

- Similarity computation with various measures (cosine, euclidean, inner product)
- Category-level retrieval evaluation
- Instance-level retrieval evaluation (image-to-text and text-to-image)
- Support for real-world datasets (currently Flickr30k, with plans to support MSCOCO)

## Usage Examples

### Computing Similarity Matrix

```python
from fico_itr import compute_similarity

similarity_matrix = compute_similarity(image_features, text_features, measure='cosine')
```

### Category-level Retrieval

```python
from fico_itr import category_retrieval

results = category_retrieval(similarity_matrix, image_labels, text_labels)
print(f"mAP: {results['mAP']:.4f}")
```

### Instance-level Retrieval

```python
from fico_itr import image_to_text_retrieval, text_to_image_retrieval

i2t_results = image_to_text_retrieval(similarity_matrix)
t2i_results = text_to_image_retrieval(similarity_matrix.T)

print(f"Image-to-Text R@1: {i2t_results[0]:.2f}%")
print(f"Text-to-Image R@1: {t2i_results[0]:.2f}%")
```

## Documentation

For more detailed information about the FiCo-ITR API, please refer to our [documentation](https://fico-itr.readthedocs.io).

## License

FiCo-ITR is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Citation

If you use FiCo-ITR in your research, please cite our paper:

```
[Citation details will be added upon paper acceptance]
```

## Contact

For any questions or issues, please open an issue on our [GitHub repository](https://github.com/MikelWL/fico-itr).

---

Remember, this is a work in progress. We appreciate your patience and feedback as we continue to improve FiCo-ITR!