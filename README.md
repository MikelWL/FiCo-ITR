# FiCo-ITR: Fine-grained and Coarse-grained Image-Text Retrieval

<p align="center">
  <a href="https://github.com/MikelWL/FiCo-ITR"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.7+-green.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-red.svg" alt="License"></a>
</p>

<p align="center">
  <b>A unified evaluation library for image-text retrieval research</b>
</p>

---

## Overview

FiCo-ITR provides a standardized evaluation framework for image-text retrieval models, supporting both instance-level (fine-grained) and category-level (coarse-grained) retrieval evaluation. The library handles diverse data formats encountered in current research, making it easy to evaluate and compare different models.

## ✨ Features

- **Universal Format Support** - Automatically handles various embedding and matrix formats and orientations
- **Flexible Distributions** - Supports both uniform and non-uniform caption distributions  
- **Dual Evaluation** - Instance-level and category-level retrieval tasks and metrics
- **Zero Configuration** - Works out-of-the-box for most models
- **Extensible** - Easy to adapt for custom evaluation needs

## 📦 Installation

```bash
pip install fico_itr
```

## 🚀 Quick Start

```python
from fico_itr import instance_retrieval, category_retrieval

# Compute Similarity Matrix
similarity_matrix = compute_similarity(image_embeddings, text_embeddings, measure='cosine')

# Instance-level retrieval - just pass your similarity matrix
i2t_instance_results, t2i_instance_results = instance_retrieval(similarity_matrix)

# Category-level retrieval - add labels for mAP evaluation
i2t_category_results, t2i_category_results = category_retrieval(similarity_matrix, labels)

print(f"Instance Retrieval Results - I2T: {i2t_instance_results} T2I: {t2i_instance_results}")
print(f"Category Retrieval Results - I2T: {i2t_category_results} T2I: {t2i_category_results}")
```

## 📖 Usage Guide

### 1. Standard Case (Uniform Images x Captions Matrix)
```python
# Just pass your similarity matrix
results = instance_retrieval(similarity_matrix)
```

### 2. Non-uniform Caption Distribution
For datasets where images have varying numbers of captions (e.g., COCO with 25010 captions):
```python
# Option A: Load pre-computed mapping (for COCO)
caption_mapping = np.load('mscoco_test_indices.npy')

# Option B: Create your own mapping
caption_mapping = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ...]  # Which image each caption belongs to

results = instance_retrieval(similarity_matrix, captions_per_image=caption_mapping)
```

### 3. Square Matrices (Duplicated Images)
Some models duplicate images to create square matrices:
```python
# E.g., vsrn/ucch with 5000×5000 matrix (1000 images duplicated 5×)
results = instance_retrieval(similarity_matrix, captions_per_image=5)
```

### 4. Separate Directional Matrices
For models with different similarity matrices per direction:
```python
# E.g., BLIP2, XVLM with task-specific fine-tuning
results = instance_retrieval(
    i2t_similarity_matrix,
    t2i_sim=t2i_similarity_matrix,
    captions_per_image=5  # or caption_mapping for non-uniform
)
```

## 📊 Model Examples

| Model | Dataset | Special Handling | Parameter |
|-------|---------|------------------|-----------|
| SCAN, IMRAM | Any | None | - |
| BEiT-3 | Flickr30k | None | - |
| BEiT-3 | COCO | Non-uniform | `caption_mapping` |
| VSRN, UCCH | Any | Square matrix | `captions_per_image=5` |
| BLIP-2, X-VLM | Any | Separate matrices | `t2i_sim=...` |
| DADH | Any | Auto-transposed | - |

## 📚 Documentation

- [**Technical Documentation**](docs/) - Implementation details
  - [Alignment](docs/alignment.md) - How data alignment works
  - [Similarity](docs/similarity.md) - Available similarity measures  
  - [Tasks](docs/tasks.md) - Evaluation metrics and algorithms
- [**Paper Tutorial**](paper_tutorial/) - Reproduce paper results with pre-computed embeddings

## 📋 Requirements

- Python 3.7+
- numpy

## 📄 Citation

If you use FiCo-ITR in your research, please cite:

```bibtex
@article{williams-lekuona2025ficoitr,
  author  = {Mikel Williams-Lekuona and Georgina Cosma},
  title   = {FiCo-ITR: Bridging Fine-Grained and Coarse-Grained Image-Text Retrieval for Comparative Performance Analysis},
  journal = {International Journal of Multimedia Information Retrieval},
  volume  = {14},
  number  = {2},
  pages   = {20},
  year    = {2025},
  publisher={Springer}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*This project includes code generated with the assistance of AI coding tools.*