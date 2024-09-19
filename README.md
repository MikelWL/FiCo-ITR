# FiCo-ITR: Fine-grained and Coarse-grained Image-Text Retrieval Library

**Note: This is the first preview 0.1.0 work in progress minimal working version. The full implementation will be released upon acceptance of the accompanying paper.**

FiCo-ITR is a Python library designed to facilitate unified evaluation of fine-grained and coarse-grained image-text retrieval models. It provides tools for computing similarity matrices and performing various retrieval tasks.

## Installation

You can install FiCo-ITR using pip:

```bash
pip install fico_itr
```

## Dependencies

FiCo-ITR requires the following Python libraries:

- numpy
- scipy (If loading embeddings for demo)

These dependencies will be automatically installed when you install FiCo-ITR using pip.

## Usage

Here's a simple example of how to use FiCo-ITR:

```python
import numpy as np
from fico_itr import compute_similarity, category_retrieval, instance_retrieval
from scipy import io

# Load your image and text embeddings. Alternatively, directly use those produced by model
image_embeddings = np.load('results_data/vsrn_f30k_img.npy')
text_embeddings = np.load('results_data/vsrn_f30k_txt.npy')
mat_data = io.loadmat('results_data/flickr30k-karpathy-test-lall.mat')
labels = mat_data['LAll']

# Compute similarity matrix
similarity_matrix = compute_similarity(image_embeddings, text_embeddings, measure='cosine')

# Perform image-to-text retrieval
i2t_instance_results, t2i_instance_results = instance_retrieval(similarity_matrix)
i2t_category_results, t2i_category_results = category_retrieval(similarity_matrix, labels)

print(f"Instance Retrieval Results: \n Image-to-Text: {i2t_instance_results} \n Text-to-Image: {t2i_instance_results}")
print(f"Category Retrieval Results: \n Image-to-Text: {i2t_category_results} \n Text-to-Image: {t2i_category_results}")
```

## Features

- Similarity computation with various measures (cosine, euclidean, inner product)
- Category-level retrieval evaluation
- Instance-level retrieval evaluation (image-to-text and text-to-image)
- Support for real-world datasets (currently Flickr30k, with plans to support MSCOCO)

## Documentation

For more detailed information about the FiCo-ITR implementation, please refer to our documentation under docs detailing issues pertaining to alignment, similarity measures, and tasks.

## Known WIP issues towards full implemenation

 - Pre-computed asymmetric matrices not yet supported.
 - Variable caption amounts per image not yet supported.

## License

FiCo-ITR is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Citation

If you use FiCo-ITR in your research, please cite our paper:

```
https://doi.org/10.48550/arXiv.2407.20114
```

## Contact

For any questions or issues, please open an issue on our [GitHub repository](https://github.com/MikelWL/fico-itr).

---